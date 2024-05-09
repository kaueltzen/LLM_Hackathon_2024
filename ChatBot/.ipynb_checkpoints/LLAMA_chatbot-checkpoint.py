from langchain_community.llms import Ollama
import streamlit as st


llm = Ollama(model="name_of_finetuned_model") # Adapt the name here as you provide it when loading locally saved gguf model

st.title("Predict max phonon peak frequency for your structure.")


# Set font size using Markdown and HTML/CSS syntax
# Add css to make text bigger
st.markdown(
    """
    <style>
    textarea {
        font-size: 1.5rem !important;
    }
    input {
        font-size: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

prompt = st.text_area("Please enter a text description of your structure", height=325)

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):

            alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n ### Instruction:\nWhat is the frequency of the highest frequency optical phonon mode peak in units of 1/cm given the following description?\n\n### Input:\n{prompt}\n\n### Response:\n
            """

            st.write_stream(llm.stream(alpaca_prompt, stop=["<|end_of_text|>"]))
