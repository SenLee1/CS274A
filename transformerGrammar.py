# transformerGrammar.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)
# The question was created by Haoyu Du (duhy@shanghaitech.edu.cn).


import util

import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, PreTrainedModel
from transformers.models.gpt_neo import GPTNeoConfig, GPTNeoForCausalLM


class InvalidTreeError(Exception):
    pass


def mapping_function(example: dict) -> dict:
    """
    Question:
        Your task is to return the processed input, processed output, attention mask, and absolute positions of the action sequence for valid actions sequence. The following order may be your implementation order:

            1. Check whether the given action sequence is a valid sequence to generate a legal parse tree. If it is invalid, please raise an InvalidTreeError Exception.
            2. The processed input: a list of strings. It should duplicate all closing nonterminals in the given action sequence.
            3. The processed output: a list of strings. It should insert '<pad>' after all closing nonterminals in the given action sequence.
            4. The absolute positions: a list of integers. The absolute position of each token is defined as the depth of it in the tree.
            5. The attention mask: a 2d torch tensor. This is the attention mask with STACK/COMPOSE attention. The attention mask of '</s>' is all 0s.

        HINT: It is guaranteed that the first item of input is '<s>' (beginning of sequence), and the last item of input is '</s>' (end of sequence). The absolute positions of both '<s>' and '</s>' are 0 in this question.
    
    Args:
        example (dict): The example to process. It has the following fields:
            - actions (List[str]): The action sequence. It is a list of strings which can be regarded as an action sequence for generative transition-based parsing.

    Return:
        mapped (dict): The mapped example. It has the following fields:
            - inputs (List[str]): The processed input. A list of tokens for the input.
            - labels (List[str]): The processed output. A list of tokens for the expected output.
            - position_ids (List[int]): The absolute positions. A list of integers representing the absolute position of each token in the input.
            - attention_mask (torch.Tensor): The attention mask. Shape: (len(input), len(input)). A 2D tensor representing the attention mask for the input sequence. 1 for valid tokens, 0 for padding tokens.

    Example:
        >>> mapping_function({"actions": ["<s>", "(S", "(NP", "the", "blue", "bird", "NP)", "(VP", "sings", "VP)", "S)", "</s>"]})
        {
            'inputs': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', 'NP)', '(VP', 'sings', 'VP)', 'VP)', 'S)', 'S)', '</s>'],
            'labels': ['<s>', '(S', '(NP', 'the', 'blue', 'bird', 'NP)', '<pad>', '(VP', 'sings', 'VP)', '<pad>', 'S)', '<pad>', '</s>'],
            'position_ids': [0, 0, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 0, 0, 0],
            'attention_mask': tensor([[...]])
        }
    """

    """YOUR CODE HERE"""
    #  1. Check whether the given action sequence is a valid sequence to generate a legal parse tree. If it is invalid, please raise an InvalidTreeError Exception.
    answer_dict = {}
    valid_OPT = ["NP", "VP", "S", "PP", "ADJP", "ADVP", "PP", "SBAR", "WHNP", "INTJ", "FRAG","UCP", "PRN","SBARQ", "SQ","QP", "<unk>","WHADVP","PRT","WHADJP","WHPP","CONJP"]
    def invalid_tree(actions):
        open_ts = []
        noterminals = True
        if actions[1][1:] not in valid_OPT:
            #  print(111111)
            return True
        for act in actions:
            if act == '<s>' or act == '</s>':
                continue
            if act.startswith('('):
                if act[1:] not in valid_OPT:
                    #  print(222222222)
                    return True
                open_ts.append(act[1:])
                noterminals = True
            elif act.endswith(')'):
                # Close a nonterminal
                if not open_ts or act[:-1] != open_ts[-1]:
                    #  print(3333333333)
                    return True
                elif noterminals:
                    #  print(44444444444444)
                    return True
                else:
                    open_ts.pop()
            else:
                noterminals = False
        if not open_ts:
            return False
        else:
            #  print(555555555555555)
            return True


    if invalid_tree(example["actions"]):
        print(example["actions"])
        raise InvalidTreeError("The action sequence is invalid to generate a legal parse tree.")


    #  2. The processed input: a list of strings. It should duplicate all closing nonterminals in the given action sequence.
    #  3. The processed output: a list of strings. It should insert '<pad>' after all closing nonterminals in the given action sequence.
    actions = example["actions"]
    inputs = []
    labels = []
    j = 1
    for act in actions:
        inputs.append(act)
        labels.append(act)
        if act.endswith(')'):
            inputs.append(act)
            labels.append('<pad>')

    #  4. The absolute positions: a list of integers. The absolute position of each token is defined as the depth of it in the tree.
    position_ids = []
    length = len(inputs)
    current_depth = 0
    for i in range(length):
        if i == 0 or i == length - 1:
            position_ids.append(0)
            continue
        if i==1 or i==len(labels)-2:
            position_ids.append(0)
            current_depth +=1
            continue
        if labels[i][0]=='(':
            position_ids.append(current_depth)
            current_depth += 1
        elif labels[i][-1]==')':
            current_depth -= 1
            position_ids.append(current_depth)
        else:
            position_ids.append(current_depth)

    #  5. The attention mask: a 2d torch tensor. This is the attention mask with STACK/COMPOSE attention. The attention mask of '</s>' is all 0s.
    attention_mask = torch.tril(torch.ones(length,length,dtype=torch.float))
    open_term = []
    #  close_term = []
    open_close_pair = []
    pad_term = []
    for i in range(length):
        if labels[i][0] == '(':
            open_term.append(i)
        elif labels[i][-1] == ')':
            open_close_pair.append((open_term[-1], i))
            open_term.pop()
        elif labels[i] == '<pad>':
            pad_term.append(i)

    # pad 下面每一行置为0
    len_pad = len(pad_term)
    for i in range(len_pad):
        attention_mask[pad_term[i]+1:,pad_term[i]] = 0

    for ont, cnt in open_close_pair:
        # Close 那一行open terminal之前置为0
        attention_mask[cnt,:ont] = 0
        # 该Open-Close对下面的所有行置为0
        attention_mask[cnt+1:length,ont:cnt] = 0

    #  for i in range(len(open_term)):
    #      for j in range(len(close_term)):
    #          if open_term[i][1][1:] == close_term[j][1][:-1]:
    #              # Close 那一行open terminal之前置为0
    #              for k in range(open_term[i][0]):
    #                  attention_mask[close_term[j][0],k] = 0
    #              # 该Open-Close对下面的所有行置为0
    #              for k in range(close_term[j][0] + 1, len(answer_dict["labels"])):
    #                  for l in range(open_term[i][0], close_term[j][0]):
    #                      attention_mask[k,l] = 0
    #              break
    # 确保最后一行全0
    attention_mask[-1, :] = 0

    answer_dict["inputs"] = inputs
    answer_dict["labels"] = labels
    answer_dict["position_ids"] = position_ids
    answer_dict["attention_mask"] = attention_mask
    return answer_dict

def get_trainer(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    train_dataset: Dataset
) -> Trainer:
    """
    Question:
        Create a Trainer object for the model. The Trainer is used to train the model on the dataset.
        Select the appropriate training arguments for the Trainer. For example, setting the proper learning rate,
        batch size, optimizer, learning rate scheduler, number of epochs, etc. would be a good idea.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use for the model.
        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The dataset to train on.

    Returns:
        trainer (Trainer): The Trainer object for the model.

    Example:
        >>> trainer = get_trainer(tokenizer, model, train_dataset)
        >>> trainer.train()
        >>> trainer.evaluate(train_dataset)
        {'eval_loss': 2.1234, ...}
    """

    def data_collator(features):
        """
        Data collator is to aggregate the features into a batch. You'll find it helpful when creating the Trainer.
        We simply pad the sequences but deal with attention mask seperately.
        """
        max_length = max([len(f["input_ids"]) for f in features])
        batch = {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "attention_mask": [],
        }
        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            position_ids = f["position_ids"]
            attention_mask = f["attention_mask"]
            seq_len = len(input_ids)

            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
            labels += [-100] * (max_length - len(labels))
            position_ids += [0] * (max_length - len(position_ids))
            attention_mask = F.pad(torch.tensor(attention_mask), [0, max_length - seq_len, 0, max_length - seq_len])

            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["position_ids"].append(position_ids)
            batch["attention_mask"].append(attention_mask)

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.stack(batch["attention_mask"])

        return batch

    """YOUR CODE HERE"""
    training_args = TrainingArguments(
        output_dir="./results",  
        learning_rate=6e-5,                        
        per_device_train_batch_size=4,             
        num_train_epochs=3,                        
        weight_decay=0.01,                         
    )
    #  training_args = TrainingArguments(
    #      output_dir="./results",
    #      evaluation_strategy="epoch",
    #      learning_rate=2e-5,
    #      per_device_train_batch_size=16,
    #      per_device_eval_batch_size=16,
    #      num_train_epochs=3,
    #      weight_decay=0.01,
    #  )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    return trainer
    #  util.raiseNotDefined()


def main():
    """This function trains a Transformer Grammar model based on GPT2 for the task of generative transition-based parsing."""
 
    ## Load the dataset from disk
    dataset = load_dataset("text", data_files="data/corpus.cc", split="train")


    ## Build the word tokenizer
    # Initialize tokenizer with special tokens
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))

    # Use the whitespace pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Build the vocabulary using WordLevelTrainer
    trainer = WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    tokenizer.train_from_iterator(dataset["text"], trainer=trainer)

    # Set the post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )

    # Convert to PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>'})


    ## Preprocess the dataset
    def tokenize_function(example):
        tokenized = tokenizer.tokenize(example["text"], add_special_tokens=True)
        return {"actions": tokenized}

    def convert_function(examples):
        input_ids = tokenizer(examples["inputs"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = tokenizer(examples["labels"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
        labels = [[(idx if idx != tokenizer.pad_token_id else -100) for idx in sent] for sent in labels]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": examples["position_ids"],
            "attention_mask": [[mask] for mask in examples["attention_mask"]],
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=["text"], load_from_cache_file=False)
    mapped_dataset = tokenized_dataset.map(mapping_function, batched=False, remove_columns=["actions"], load_from_cache_file=False)
    converted_dataset = mapped_dataset.map(convert_function, batched=True, remove_columns=["inputs"], load_from_cache_file=False)


    # Load the model
    # TODO: use GPT2 instead of GPTNeo when transformers 4.52.0 is released
    # We use GPTNeo here since the implementation of GPT2 has a bug and the fix has not been released yet.
    # GPTNeo is similar to GPT2 except that it uses local attention. We have disabled local attention in the config.
    config = GPTNeoConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_layers=6,
        num_heads=8,
        attention_types=[[["global"], 6]],
        activation_function="relu",
    )
    model = GPTNeoForCausalLM(config)


    # Training
    trainer = get_trainer(tokenizer, model, converted_dataset)
    trainer.train()
    metrics = trainer.evaluate(converted_dataset)

    print(metrics)


if __name__ == "__main__":
    main()
