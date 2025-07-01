
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
#
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Callable
import argparse
from transformers import pipeline
import gradio as gr


def get_topic_classification_pipeline() -> Callable[[str], dict]:
    """
    Question:
        Load the pipeline for topic text classification.
        There are 10 possible labels: 
            'Society & Culture', 'Science & Mathematics', 'Health',
            'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance',
            'Entertainment & Music', 'Family & Relationships', 'Politics & Government'
        Find a proper model from HuggingFace Model Hub, then load the pipeline to classify the text.
        Notice that we have time limits so you should not use a model that is too large. A model with 
        100M params is enough.

    Returns:
        func (Callable): A function that takes a string as input and returns a dictionary with the
        predicted label and its score.

    Example:
        >>> func = get_topic_classification_pipeline()
        >>> result = func("Would the US constitution be changed if the admendment received 2/3 of the popular vote?")
        {"label": "Politics & Government", "score": 0.9999999403953552}
    """
from typing import Callable
from transformers import pipeline

def get_topic_classification_pipeline() -> Callable[[str], dict]:
    """
    Load the pipeline for topic text classification.
    There are 10 possible labels: 
        'Society & Culture', 'Science & Mathematics', 'Health',
        'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance',
        'Entertainment & Music', 'Family & Relationships', 'Politics & Government'
    Find a proper model from HuggingFace Model Hub, then load the pipeline to classify the text.
    Notice that we have time limits so you should not use a model that is too large. A model with 
    100M params is enough.

    Returns:
        func (Callable): A function that takes a string as input and returns a dictionary with the
        predicted label and its score.

    Example:
        >>> func = get_topic_classification_pipeline()
        >>> result = func("Would the US constitution be changed if the admendment received 2/3 of the popular vote?")
        original code:
            pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
            def func(text: str) -> dict:
                return pipe(text)[0]
            return func

        python classification.py --task topic
        {"label": "Politics & Government", "score": 0.9999999403953552}
    """
    candidate_labels = [
            'Society & Culture', 'Science & Mathematics', 'Health',
            'Education & Reference', 'Computers & Internet', 'Sports',
            'Business & Finance', 'Entertainment & Music',
            'Family & Relationships', 'Politics & Government'
    ]

    pipe = pipeline(
        "zero-shot-classification",
        #  model="typeform/distilbert-base-uncased-mnli",  2':<0.4
        #  model="cross-encoder/nli-MiniLM2-L6-H768" 2':0.4
        model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33" # 3':0.51
    )
    
    def func(text: str) -> dict:
        result = pipe(
            text,
            candidate_labels=candidate_labels,
            hypothesis_template="This text is mainly about {}."  ,
        )
        return {"label": result["labels"][0], "score": result["scores"][0]}
    
    return func
def main():
    parser = argparse.ArgumentParser(
        description="Topic Classification Pipeline")
    parser.add_argument("--task", type=str, help="Task name",
                        choices=["sentiment", "topic"], default="sentiment")
    parser.add_argument("--use-gradio", action="store_true",
                        help="Use Gradio for UI")

    args = parser.parse_args()

    if args.use_gradio and args.task == "sentiment":
        # Example usage with Gradio
        pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
        iface = gr.Interface.from_pipeline(pipe)
        iface.launch()

    elif args.use_gradio and args.task == "topic":
        # Visualize the topic classification pipeline with Gradio
        pipe = get_topic_classification_pipeline()
        iface = gr.Interface(
            fn=lambda x: {item["label"]: item["score"] for item in [pipe(x)]},
            inputs=gr.components.Textbox(label="Input", render=False),
            outputs=gr.components.Label(label="Classification", render=False),
            title="Text Classification",
        )
        iface.launch()

    elif not args.use_gradio and args.task == "sentiment":
        # Example usage
        pipe = pipeline(model="cointegrated/rubert-tiny-sentiment-balanced")
        # {'label': 'positive', 'score': 0.988831102848053}
        print(pipe("This movie is great!")[0])

    elif not args.use_gradio and args.task == "topic":
        # Test the function
        func = get_topic_classification_pipeline()
        # {"label": "Politics & Government", "score": ...}
        print(func("Would the US constitution be changed if the admendment received 2/3 of the popular vote?"))


if __name__ == "__main__":
    main()
