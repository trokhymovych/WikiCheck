import time
import json
from argparse import ArgumentParser
from datetime import datetime, timedelta
from typing import Annotated, List
from urllib.parse import urljoin
from uuid import uuid4
from pathlib import Path

import uvicorn

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)

from modules.entities import (
    TextRequest,
    Token,
    User,
    NLIRequest,
    NLIResponse,
    FactCheckResponse,
    FactCheckNOResponse
)

from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from modules.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    update_user_credits,
    validate_user,
)
from modules.constants import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    TEMPLATES_DIRECTORY,
)
from modules.model_complex import WikiFactChecker
from modules.utils.logging_utils import get_logger, check_if_none, ROOT_LOGGER_NAME, DataLoggerFile


parser = ArgumentParser()
parser.add_argument('--config', type=str, required=False,
                    default='configs/inference/sentence_bert_config.json', help='path to config')

args = parser.parse_args()
config_path = args.config

logger = get_logger(name=ROOT_LOGGER_NAME,
                    console=True,
                    log_level="INFO",
                    propagate=False)

logger.info(f"Reading config from {Path(config_path).absolute()}")
with open(config_path) as con_file:
    config = json.load(con_file)
logger.info(f"Using config {config}")


logger.info(f"Loading models ...")
# Instantiate models classes
complex_model = WikiFactChecker(config, logger=logger)
logger.info(f"Models loaded.")

# Instantiate logging classes
data_logger = DataLoggerFile(file_path=config.get("data_logs_path"))

# instantiate API class
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

print(config)

# define information to show on default page
templates = Jinja2Templates(directory=TEMPLATES_DIRECTORY)
api_title = config.get("title")
api_description = config.get("description")
api_version = config.get("version")
public_metadata = {
    "version": api_version,
}
app = FastAPI(title=api_title, description=api_description, version=api_version)



@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = round(time.time() - start_time, 4)
    response.headers["X-Process-Time"] = str(process_time)

    body = b""
    if hasattr(response, "body"):
        body = response.body
    elif hasattr(response, "body_iterator"):
        async for chunk in response.body_iterator:
            if not isinstance(chunk, bytes):
                chunk = chunk.encode(response.charset)
            body += chunk
    content = body.decode("utf-8")

    data_log = {
        "id": response.headers.get("X-request-ID", "unknown"),
        "path": request.url.path,
        "response": content,
        "user": response.headers.get("X-user", "unknown"),
        "time": datetime.now().strftime("%Y-%m-%d, %H:%M"),
        "process_time": process_time,
        "status_code": response.status_code,
    }
    task = BackgroundTask(data_logger.write, data_log)
    return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            background=task
        )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    base_url = str(request.base_url)
    docs_link = urljoin(base_url, "docs")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": api_title,
            "description": api_description,
            "metadata": public_metadata,
            "docs_link": docs_link,
        },
    )


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return Token(access_token=access_token, token_type="bearer")


@app.post("/get-nli-prediction/", tags=["NLI"])
async def get_nli_prediction(
    nli_input: NLIRequest,
    response: Response,
    background_tasks: BackgroundTasks,
) -> NLIResponse:
    response.headers["X-request-ID"] = str(uuid4())
    # Step 1: Process input data:
    claim = check_if_none(nli_input.claim)
    hypothesis = check_if_none(nli_input.hypothesis)
    try:
        # Step 2: Prediction:
        result = complex_model.model_level_two.predict(claim, hypothesis)
        result["request"] = str({"text": claim, "hypothesis": hypothesis})
        return result
    except Exception as e:
        data_log = {
            "id": response.headers["X-request-ID"],
            "path": "/get-nli-prediction/",
            "reason": str(e),
            "time": datetime.now().strftime("%Y-%m-%d, %H:%M"),
            "request": str({"text": claim, "hypothesis": hypothesis}),
            "response": None,
            "user": "None",
        }
        background_tasks.add_task(data_logger.write, data=data_log)
        raise HTTPException(
            status_code=500,
            detail="Service internal error. Method get-nli-prediction failed.",
            headers={"X-Error": "Service internal error"},
        )
    

@app.post("/get-fact-check-non-aggregated/", tags=["Fact checking"])
async def get_fact_check_non_aggregated(
    claim: TextRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(validate_user)],
) -> FactCheckNOResponse: 
    response.headers["X-request-ID"] = str(uuid4())
    response.headers["X-user"] = current_user.username
    claim_text = check_if_none(claim.text)
    try:
        result = complex_model.predict_all(claim_text)
        result_final = {
            "request": claim_text,
            "predicted_evidence": result,
        }
        # Update user credits
        update_user_credits(current_user.username, -1)
        return result_final
    except Exception as e:
        data_log = {
            "id": response.headers["X-request-ID"],
            "reason": str(e),
            "path": "/get-fact-check-non-aggregated/",
            "time": datetime.now().strftime("%Y-%m-%d, %H:%M"),
            "request": claim_text,
            "response": None,
            "user": current_user.username,
        }
        background_tasks.add_task(data_logger.write, data=data_log)
        raise HTTPException(
            status_code=500,
            detail="Service internal error. Method get-classification failed.",
            headers={"X-Error": "Service internal error"},
        )


@app.post("/get-fact-check-aggregated-base/", tags=["Fact checking"])
async def get_fact_check_aggregated_base(
    claim: TextRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(validate_user)],
) -> FactCheckResponse:
    response.headers["X-request-ID"] = str(uuid4())
    response.headers["X-user"] = current_user.username
    claim_text = check_if_none(claim.text)
    try:
        result = complex_model.predict_and_aggregate(claim_text)
        result["request"] = claim_text
        # Update user credits
        update_user_credits(current_user.username, -1)
        return result
    except Exception as e:
        data_log = {
            "id": response.headers["X-request-ID"],
            "reason": str(e),
            "path": "/get-fact-check-aggregated-base/",
            "time": datetime.now().strftime("%Y-%m-%d, %H:%M"),
            "request": claim_text,
            "response": None,
            "user": current_user.username,
        }
        background_tasks.add_task(data_logger.write, data=data_log)
        raise HTTPException(
            status_code=500,
            detail="Service internal error. Method get-classification failed.",
            headers={"X-Error": "Service internal error"},
        )


# Get method to get the users details:
@app.get("/me", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    return current_user


if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=config.get("port", 80), log_level="info", reload=False)
