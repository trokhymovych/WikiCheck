import sys
import json
from argparse import ArgumentParser
from pathlib import Path
sys.path.insert(0, "/Users/ntr/Documents/tresh/fairapi")
# sys.path.insert(0, "/home/trokhymovych/fairapi")

import streamlit as st
from annotated_text import annotated_text
from colour import Color


@st.cache(allow_output_mutation=True)
def get_converters():
    from modules.model_complex import WikiFactChecker
    from modules.utils.logging_utils import get_logger, ROOT_LOGGER_NAME

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=False,
                        default='modules/configs/sentence_bert_config.json', help='path to config')

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

    complex_model = WikiFactChecker(logger, **config)

    return complex_model


def write_header():
    st.title('Fact checking system by Wikipedia')
    st.markdown('''
        - That is a system that makes a fact verification for a given claim and interpret final results.
    ''')


def parse_results(res):
    articles = dict()
    for r in res:
        current_text = articles.get(r['article'], [])
        po_p = r['entailment_prob']
        ne_p = r['contradiction_prob']
        if po_p > 0.5:
            current_text.append((r['text']+'. ', "SUPPORTS", Color("green", luminance=po_p/2).get_hex_l()))
        elif ne_p > 0.5:
            current_text.append((r['text']+'. ', "REFUTES", Color("red", luminance=ne_p/2).get_hex_l()))
        else:
            current_text.append(r['text']+'. ')
        articles[r['article']] = current_text
    return articles


def write_ui():
    claim = st.text_input('Enter English sentence below and hit Enter', value="Moscow is a capital of Ukraine")
    if not claim:
        return
    complex_model = get_converters()
    results = complex_model.predict_all(claim)
    articles = parse_results(results)

    for k, v in articles.items():
        st.markdown(f''' #### Article name: {k.replace('_', ' ')}''')
        annotated_text(v)


def write_footer():
    st.markdown('''
    ''')


def production_mode():
    # Src: discuss.streamlit.io/t/how-do-i-hide-remove-the-menu-in-production/362
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    return


if __name__ == '__main__':
    st.set_page_config(page_title='Fact checking system by Wikipedia', page_icon='⚖', layout='wide')
    production_mode()
    write_header()
    write_ui()
    write_footer()
