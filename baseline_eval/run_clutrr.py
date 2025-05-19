import os

from argparse import ArgumentParser
from tqdm import tqdm
import csv
import re
import random
import transformers

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
import math
import ast

# import scallopy

relation_id_map = {
  'daughter': 0,
  'sister': 1,
  'son': 2,
  'aunt': 3,
  'father': 4,
  'husband': 5,
  'granddaughter': 6,
  'brother': 7,
  'nephew': 8,
  'mother': 9,
  'uncle': 10,
  'grandfather': 11,
  'wife': 12,
  'grandmother': 13,
  'niece': 14,
  'grandson': 15,
  'son-in-law': 16,
  'father-in-law': 17,
  'daughter-in-law': 18,
  'mother-in-law': 19,
  'nothing': 20,
}

# class ClutrrDataset:
#   def __init__(self, train=False, varied_complexity=False, root="./"):
#     self.dataset_dir = os.path.join(root, f"data/CLUTRR/")
#     split = "train" if train else "test"
#     self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if "clutrr_" in d]
#     print(self.file_names)
#     self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]
#     self.data_num = math.floor(len(self.data) * 100 / 100)
#     self.data = self.data[:self.data_num]

#   def __len__(self):
#     return len(self.data)

#   def __getitem__(self, i):
#     # Context is a list of sentences
#     context = [s.strip().lower() for s in self.data[i][2].split(".") if s.strip() != ""]

#     # Query is of type (sub, obj)
#     query_sub_obj = eval(self.data[i][3])
#     query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

#     # Answer is one of 20 classes such as daughter, mother, ...
#     answer = self.data[i][5]
#     return ((context, query), answer)

#   @staticmethod
#   def collate_fn(batch):
#     queries = [query for ((_, query), _) in batch]
#     contexts = [fact for ((context, _), _) in batch for fact in context]
#     context_lens = [len(context) for ((context, _), _) in batch]
#     context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
#     answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer) in batch])
#     print(contexts, queries, context_splits)
#     return ((contexts, queries, context_splits), answers)
def function(facts, query):
  rules = {
     ("daughter", "daughter"): "granddaughter",
        ("daughter", "sister"): "daughter",
        ("daughter", "son"): "grandson",
        ("daughter", "aunt"): "sister",
        ("daughter", "father"): "husband",
        ("daughter", "husband"): "son-in-law",
        ("daughter", "brother"): "son",
        ("daughter", "mother"): "wife",
        ("daughter", "uncle"): "brother",
        ("daughter", "grandfather"): "father",
        ("daughter", "grandfather"): "father-in-law",
        ("daughter", "grandmother"): "mother",
        ("daughter", "grandmother"): "mother-in-law",
        ("sister", "daughter"): "niece",
        ("sister", "sister"): "sister",
        ("sister", "son"): "nephew",
        ("sister", "aunt"): "aunt",
        ("sister", "father"): "father",
        ("sister", "brother"): "brother",
        ("sister", "mother"): "mother",
        ("sister", "uncle"): "uncle",
        ("sister", "grandfather"): "grandfather",
        ("sister", "grandmother"): "grandmother",
        ("son", "daughter"): "granddaughter",
        ("son", "sister"): "daughter",
        ("son", "son"): "grandson",
        ("son", "aunt"): "sister",
        ("son", "father"): "husband",
        ("son", "brother"): "son",
        ("son", "mother"): "wife",
        ("son", "uncle"): "brother",
        ("son", "grandfather"): "father",
        ("son", "grandfather"): "father-in-law",
        ("son", "grandmother"): "mother",
        ("son", "grandmother"): "mother-in-law",
        ("aunt", "sister"): "aunt",
        ("aunt", "father"): "grandfather",
        ("aunt", "brother"): "uncle",
        ("aunt", "mother"): "grandmother",
        ("father", "daughter"): "sister",
        ("father", "sister"): "aunt",
        ("father", "son"): "brother",
        ("father", "father"): "grandfather",
        ("father", "brother"): "uncle",
        ("father", "mother"): "grandmother",
        ("father", "wife"): "mother",
        ("husband", "daughter"): "daughter",
        ("husband", "son"): "son",
        ("husband", "father"): "father-in-law",
        ("husband", "granddaughter"): "granddaughter",
        ("husband", "mother"): "mother-in-law",
        ("husband", "grandson"): "grandson",
        ("granddaughter", "sister"): "granddaughter",
        ("granddaughter", "brother"): "grandson",
        ("brother", "daughter"): "niece",
        ("brother", "sister"): "sister",
        ("brother", "son"): "nephew",
        ("brother", "aunt"): "aunt",
        ("brother", "father"): "father",
        ("brother", "brother"): "brother",
        ("brother", "mother"): "mother",
        ("brother", "uncle"): "uncle",
        ("brother", "grandfather"): "grandfather",
        ("brother", "grandmother"): "grandmother",
        ("nephew", "sister"): "niece",
        ("nephew", "brother"): "nephew",
        ("mother", "daughter"): "sister",
        ("mother", "sister"): "aunt",
        ("mother", "son"): "brother",
        ("mother", "father"): "grandfather",
        ("mother", "husband"): "father",
        ("mother", "brother"): "uncle",
        ("mother", "mother"): "grandmother",
        ("mother", "father"): "grandfather",
        ("mother", "mother"): "grandmother",
        ("uncle", "sister"): "aunt",
        ("uncle", "father"): "grandfather",
        ("uncle", "brother"): "uncle",
        ("uncle", "mother"): "grandmother",
        ("grandfather", "wife"): "grandmother",
        ("wife", "daughter"): "daughter",
        ("wife", "son"): "son",
        ("wife", "father"): "father-in-law",
        ("wife", "granddaughter"): "granddaughter",
        ("wife", "mother"): "mother-in-law",
        ("wife", "grandson"): "grandson",
        ("wife", "son-in-law"): "son-in-law",
        ("wife", "father-in-law"): "father",
        ("wife", "daughter-in-law"): "daughter-in-law",
        ("wife", "mother-in-law"): "mother",
        ("grandmother", "husband"): "grandfather",
        ("grandson", "sister"): "granddaughter",
        ("grandson", "brother"): "grandson", 
  }

  last_facts = {}
  while query not in facts:
      added_facts = {}
      for fact1 in facts.items():
          for fact2 in facts.items():
              if fact1[0][0] != fact2[0][1] and fact1[0][1] == fact2[0][0] and (fact2[1], fact1[1]) in rules and (fact1[0][0], fact2[0][1]) not in facts:
                  new_fact = rules[(fact2[1], fact1[1])]
                  added_facts[(fact1[0][0], fact2[0][1])] = new_fact
      facts.update(added_facts)
      if last_facts == facts:
          break
      last_facts = facts.copy()
  print("final facts:", facts)

  if query in facts:
      return facts[query]
  else:
      return "Uncertain"


class ClutrrDataset(torch.utils.data.Dataset):
  def __init__(self, train=False, varied_complexity=False, root="./"):
      # load jsonlines
      # self.data = load_dataset("CLUTRR/v1", "gen_train234_test2to10", split="test").to_list()
      self.data = []
      # with open("data/CLUTRR/test.jsonl", "r") as f:
      #     for line in f:
      #         self.data.append(json.loads(line))
      # load from csv
      if varied_complexity:
          for comp in range(4, 11):
              with open(root + f"data/CLUTRR/clutrr_{comp}.csv", "r") as f:
                  # read the first line to get the keys
                  reader = csv.DictReader(f)
                  for row in reader:
                      self.data.append({"question": row['story'], "answer": row['target'], "query": row['query'], "complexity": comp})
          # shuffle
          np.random.seed(0)
          self.data = np.random.permutation(self.data)
      else:
          with open(root + f"data/CLUTRR/clutrr_4.csv", "r") as f:
              # read the first line to get the keys
              reader = csv.DictReader(f)
              for row in reader:
                  self.data.append({"question": row['story'], "answer": row['target'], "query": row['query'], "complexity": 4})

      # subsample to 300
      print("Number of samples:", len(self.data))

      if not train and not varied_complexity:
          self.data = self.data[:100]
      elif not train and varied_complexity:
          self.data = self.data[:400]
      elif train and varied_complexity:
          self.data = self.data[400:]
      else:
          # get remaining samples
          self.data = self.data[100:]
      # random.seed(0)
      # self.data = random.sample(self.data, 300)

      # num_people = [len(d["genders"].split(",")) for d in self.data]
      # num_people = [len(d["name_map"]) for d in self.data]
      # # print histogram as text
      # print("Number of people histogram:")
      # print(np.histogram(num_people))
      print("Complexity histogram:")
      print(np.histogram([d["complexity"] for d in self.data], bins=range(4, 12)))

  def __getitem__(self, index):
      # story = self.data[index]["question"].split("\n")[0]
      story = self.data[index]["question"]
      context = [s.strip() for s in story.split(".") if s.strip() != ""]
      query = ast.literal_eval(self.data[index]["query"])[::-1]
      context_split = [(0, len(context))]
      # query = self.data[index]["question"].split("\n")[1]
      # get the two names in [] from the query as a tuple
      # query = str(re.findall(r"\[(.*?)\]", query))

      # return (story, query), self.data[index]["answer"].split("#### ")[1]
      print(context, query, context_split)
      return (context, query, context_split), self.data[index]["answer"]

  def __len__(self):
      return len(self.data)


def clutrr_loader(root):
  train_dataset = ClutrrDataset(train=True, root=root)

  np.random.seed(0)
  test_dataset = ClutrrDataset(train=False, root=root, varied_complexity=False)

  test_data_ids = list(range(min(200, len(test_dataset)))) #+ list(range(103, len(data)))
  shuf = np.random.permutation(test_data_ids)
  test_dataset = [test_dataset[int(i)] for i in shuf[:200]]

  # test_dataset = DataLoader(test_dataset, 1, collate_fn=ClutrrDataset.collate_fn, shuffle=False)

  return (train_dataset, test_dataset)


class MLP(nn.Module):
  def __init__(self, in_dim: int, embed_dim: int, out_dim: int, num_layers: int = 0, softmax = False, normalize = False, sigmoid = False):
    super(MLP, self).__init__()
    layers = []
    layers += [nn.Linear(in_dim, embed_dim), nn.ReLU()]
    for _ in range(num_layers):
      layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
    layers += [nn.Linear(embed_dim, out_dim)]
    self.model = nn.Sequential(*layers)
    self.softmax = softmax
    self.normalize = normalize
    self.sigmoid = sigmoid

  def forward(self, x):
    x = self.model(x)
    if self.softmax: x = nn.functional.softmax(x, dim=1)
    if self.normalize: x = nn.functional.normalize(x)
    if self.sigmoid: x = torch.sigmoid(x)
    return x


class CLUTRRModel(nn.Module):
  def __init__(
    self,
    device="cpu",
    num_mlp_layers=1,
    debug=False,
    no_fine_tune_roberta=False,
    use_softmax=False,
    provenance="difftopbottomkclauses",
    train_top_k=3,
    test_top_k=3,
  ):
    super(CLUTRRModel, self).__init__()

    # Options
    self.device = device
    self.debug = debug
    self.no_fine_tune_roberta = no_fine_tune_roberta

    # # Roberta as embedding extraction model
    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=False, add_prefix_space=True)
    self.roberta_model = RobertaModel.from_pretrained("roberta-base")
    self.embed_dim = self.roberta_model.config.hidden_size

    # Entity embedding
    self.relation_extraction = MLP(self.embed_dim * 3, self.embed_dim, len(relation_id_map), num_layers=num_mlp_layers, sigmoid=not use_softmax, softmax=use_softmax)

    # # Scallop reasoning context
    # self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, train_k=train_top_k, test_k=test_top_k)
    # self.scallop_ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), f"../clutrr.scl")))
    # self.scallop_ctx.set_non_probabilistic(["question"])

    # self.reason = self.scallop_ctx.forward_function(output_mappings={"answer": list(range(len(relation_id_map)))}) #, retain_graph=True)

  def _preprocess_contexts(self, contexts, context_splits):
    clean_context_splits = []
    clean_contexts = []
    name_token_indices_maps = []
    for (_, (start, end)) in enumerate(context_splits):
      skip_next = False
      skip_until = 0
      curr_clean_contexts = []
      curr_name_token_indices_maps = []
      for (j, sentence) in zip(range(start, end), contexts[start:end]):
        # It is possible to skip a sentence because the previous one includes the current one.
        if skip_next:
          if j >= skip_until:
            skip_next = False
          continue

        # Get all the names of the current sentence
        names = re.findall("\\[(\w+)\\]", sentence)

        # Check if we need to include the next sentence(s) as well
        num_sentences = 1
        union_sentence = f"{sentence}"
        for k in range(j + 1, end):
          next_sentence = contexts[k]
          next_sentence_names = re.findall("\\[(\w+)\\]", next_sentence)
          if len(names) == 1 or len(next_sentence_names) == 1:
            if len(next_sentence_names) > 0:
              num_sentences += 1
              union_sentence += f". {next_sentence}"
              names += next_sentence_names
            skip_next = True
            if len(next_sentence_names) == 1:
              skip_until = k - 1
            else:
              skip_until = k
          else:
            break

        # Deduplicate the names
        names = set(names)

        # Debug number of sentences
        if self.debug and num_sentences > 1:
          print(f"number of sentences: {num_sentences}, number of names: {len(names)}; {names}")
          print("Sentence:", union_sentence)

        # Then split the context by `[` and `]` so that names are isolated in its own string
        splitted = [u.strip() for t in union_sentence.split("[") for u in t.split("]") if u.strip() != ""]

        # Get the ids of the name in the `splitted` array
        is_name_ids = {s: [j for (j, sp) in enumerate(splitted) if sp == s] for s in names}

        # Get the splitted input_ids
        splitted_input_ids_raw = self.tokenizer(splitted).input_ids
        splitted_input_ids = [ids[:-1] if j == 0 else ids[1:] if j == len(splitted_input_ids_raw) - 1 else ids[1:-1] for (j, ids) in enumerate(splitted_input_ids_raw)]
        index_counter = 0
        splitted_input_indices = []
        for (j, l) in enumerate(splitted_input_ids):
          begin_offset = 1 if j == 0 else 0
          end_offset = 1 if j == len(splitted_input_ids) - 1 else 0
          quote_s_offset = 1 if "'s" in splitted[j] and splitted[j].index("'s") == 0 else 0
          splitted_input_indices.append(list(range(index_counter + begin_offset, index_counter + len(l) - end_offset - quote_s_offset)))
          index_counter += len(l) - quote_s_offset

        # Get the token indices for each name
        name_token_indices = {s: [k for phrase_id in is_name_ids[s] for k in splitted_input_indices[phrase_id]] for s in names}

        # Clean up the sentence and add it to the batch
        clean_sentence = union_sentence.replace("[", "").replace("]", "")

        # Preprocess the context
        curr_clean_contexts.append(clean_sentence)
        curr_name_token_indices_maps.append(name_token_indices)

      # Add this batch into the overall list; record the splits
      curr_size = len(curr_clean_contexts)
      clean_context_splits.append((0, curr_size) if len(clean_context_splits) == 0 else (clean_context_splits[-1][1], clean_context_splits[-1][1] + curr_size))
      clean_contexts += curr_clean_contexts
      name_token_indices_maps += curr_name_token_indices_maps

    print("name idxs:", name_token_indices_maps)

    # Return the preprocessed contexts and splits
    return (clean_contexts, clean_context_splits, name_token_indices_maps)

  def _extract_relations(self, clean_contexts, clean_context_splits, name_token_indices_maps):
    # Use RoBERTa to encode the contexts into overall tensors
    context_tokenized_result = self.tokenizer(clean_contexts, padding=True, return_tensors="pt")
    context_input_ids = context_tokenized_result.input_ids.to(self.device)
    context_attention_mask = context_tokenized_result.attention_mask.to(self.device)
    encoded_contexts = self.roberta_model(context_input_ids, context_attention_mask)
    if self.no_fine_tune_roberta:
      roberta_embedding = encoded_contexts.last_hidden_state.detach()
    else:
      roberta_embedding = encoded_contexts.last_hidden_state

    # Extract features corresponding to the names for each context
    splits, name_pairs, name_pairs_features = [], [], []

    for (begin, end) in clean_context_splits:
      curr_datapoint_name_pairs = []
      curr_datapoint_name_pairs_features = []
      curr_sentence_rep = []

      for (j, name_token_indices) in zip(range(begin, end), name_token_indices_maps[begin:end]):
        # Generate the feature_maps
        feature_maps = {}
        curr_sentence_rep.append(torch.mean(roberta_embedding[j, :sum(context_attention_mask[j]), :], dim=0))
        for (name, token_indices) in name_token_indices.items():
          token_features = roberta_embedding[j, token_indices, :]

          # Use max pooling to join the features
          agg_token_feature = torch.max(token_features, dim=0).values
          feature_maps[name] = agg_token_feature

        # Generate name pairs
        names = list(name_token_indices.keys())
        curr_sentence_name_pairs = [(m, n) for m in names for n in names if m != n]
        curr_datapoint_name_pairs += curr_sentence_name_pairs
        curr_datapoint_name_pairs_features += [torch.cat((feature_maps[x], feature_maps[y])) for (x, y) in curr_sentence_name_pairs]

      global_rep = torch.mean(torch.stack(curr_sentence_rep), dim=0)

      # Generate the pairs for this datapoint
      num_name_pairs = len(curr_datapoint_name_pairs)
      splits.append((0, num_name_pairs) if len(splits) == 0 else (splits[-1][1], splits[-1][1] + num_name_pairs))
      name_pairs += curr_datapoint_name_pairs
      name_pairs_features += curr_datapoint_name_pairs_features

    # Stack all the features into the same big tensor
    name_pairs_features = torch.cat((torch.stack(name_pairs_features), global_rep.repeat(len(name_pairs_features), 1)), dim=1)

    # Use MLP to extract relations between names
    name_pair_relations = self.relation_extraction(name_pairs_features)

    # Return the extracted relations and their corresponding symbols
    return (splits, name_pairs, name_pair_relations)

  def _extract_facts(self, splits, name_pairs, name_pair_relations, queries):
    context_facts, context_disjunctions, question_facts = [], [], []
    num_pairs_processed = 0

    # Generate facts for each context
    for (i, (begin, end)) in enumerate(splits):
      # First combine the name_pair features if there are multiple of them, using max pooling
      name_pair_to_relations_map = {}
      for (j, name_pair) in zip(range(begin, end), name_pairs[begin:end]):
        name_pair_to_relations_map.setdefault(name_pair, []).append(name_pair_relations[j])
      name_pair_to_relations_map = {k: torch.max(torch.stack(v), dim=0).values for (k, v) in name_pair_to_relations_map.items()}

      # Generate facts and disjunctions
      curr_context_facts = []
      curr_context_disjunctions = []
      for ((sub, obj), relations) in name_pair_to_relations_map.items():
        curr_context_facts += [(relations[k], (k, sub, obj)) for k in range(len(relation_id_map))]
        curr_context_disjunctions.append(list(range(len(curr_context_facts) - 20, len(curr_context_facts))))
      context_facts.append(curr_context_facts)
      context_disjunctions.append(curr_context_disjunctions)
      question_facts.append([queries[i]])

      # Increment the num_pairs processed for the next datapoint
      num_pairs_processed += len(name_pair_to_relations_map)

    # Return the facts generated
    return (context_facts, context_disjunctions, question_facts)

  def forward(self, x, phase='train'):
    (contexts, queries, context_splits) = x

    # Debug prints
    if self.debug:
      print(contexts)
      print(queries)

    # Go though the preprocessing, RoBERTa model forwarding, and facts extraction steps
    (clean_contexts, clean_context_splits, name_token_indices_maps) = self._preprocess_contexts(contexts, context_splits)
    (splits, name_pairs, name_pair_relations) = self._extract_relations(clean_contexts, clean_context_splits, name_token_indices_maps)
    (context_facts, context_disjunctions, question_facts) = self._extract_facts(splits, name_pairs, name_pair_relations, queries)

    # Run Scallop to reason the result relation
    # result = self.reason(context=context_facts, question=question_facts, disjunctions={"context": context_disjunctions})
    facts = [(names, relations.cpu()) for names, relations in zip(name_pairs, name_pair_relations)]

    # Return the final result
    return None, [facts, question_facts]
    # return result, [result, question_facts]

class Trainer:
  def __init__(self, train_loader, test_loader, device, model_dir, model_name, learning_rate, **args):
    self.device = device
    load_model = args.pop('load_model')
    if load_model:
      new_model = CLUTRRModel(device=device, **args).to(device)
      # new_model.tokenizer = torch.load(model_dir + "tok.best.model", weights_only=False)
      new_model.roberta_model = torch.load(model_dir + "roberta.best.model", weights_only=False).cpu()
      # replace all the weights in the model with the loaded weights
      # new_model.roberta_model.load_state_dict(roberta_model.state_dict())
      relation_extraction = torch.load(model_dir + "relation_extraction.best.model", weights_only=False)
      new_model.relation_extraction.load_state_dict(relation_extraction.state_dict())
      self.model = new_model
    else:
      self.model = CLUTRRModel(device=device, **args).to(device)
    self.model_dir = model_dir
    self.model_name = model_name
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.min_test_loss = 10000000000.0
    self.max_accu = 0

  def loss(self, y_pred, y):
    result = y_pred
    (_, dim) = result.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y]).to(self.device)
    result_loss = nn.functional.binary_cross_entropy(result, gt)
    return result_loss

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    result = y_pred.detach()
    pred = torch.argmax(result, dim=1)
    num_correct = len([() for i, j in zip(pred, y) if i == j])
    return (num_correct, batch_size)

  def train(self, num_epochs):
    for i in range(1, num_epochs + 1):
      self.train_epoch(i)
      self.test_epoch(i)

  def train_epoch(self, epoch):
    self.model.train()
    total_count = 0
    total_correct = 0
    total_loss = 0
    iterator = tqdm(self.train_loader)
    for (i, (x, y)) in enumerate(iterator):
      self.optimizer.zero_grad()
      y_pred = self.model(x, 'train')
      loss = self.loss(y_pred, y)
      total_loss += loss.item()
      loss.backward()
      self.optimizer.step()

      (num_correct, batch_size) = self.accuracy(y_pred, y)
      total_count += batch_size
      total_correct += num_correct
      correct_perc = 100. * total_correct / total_count
      avg_loss = total_loss / (i + 1)

      iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")

    return avg_loss, correct_perc

  def test_epoch(self, epoch):
    self.model.eval()
    total_count = 0
    total_correct = 0
    total_loss = 0
    outputs = []
    relations = [
  'daughter',
  'sister',
  'son',
  'aunt',
  'father',
  'husband',
  'granddaughter',
  'brother',
  'nephew',
  'mother',
  'uncle',
  'grandfather',
  'wife',
  'grandmother',
  'niece',
  'grandson',
  'son-in-law',
  'father-in-law',
  'daughter-in-law',
  'mother-in-law',
  'nothing',
]
    with torch.no_grad():
      iterator = tqdm(self.test_loader)
      for (i, (x, y)) in enumerate(iterator):
        y_pred, intermediate = self.model(x, 'test')
        # pred, _ = self.model(x, 'test')
        outputs.append(intermediate)

        facts = {}
        for names, rel in intermediate[0]:
            if relations[torch.argmax(rel).item()] != "nothing":
                facts[names[::-1]] = relations[torch.argmax(rel).item()]

        pred = function(facts, x[1])
        print(pred, y)

        # (num_correct, batch_size) = self.accuracy(y_pred, y)
        total_count += 1
        total_correct += 1 if pred == y else 0
        correct_perc = 100. * total_correct / total_count
        avg_loss = total_loss / (i + 1)

        iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)")

    torch.save(outputs, "clutrr_outputs.pkl")

    return avg_loss, correct_perc

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--model-name", type=str, default=None)
  parser.add_argument("--training_data_percentage", type=int, default=100)
  parser.add_argument("--load_model", type=bool, default=True)
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--learning-rate", type=float, default=0.000005)
  parser.add_argument("--num-mlp-layers", type=int, default=2)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--train-top-k", type=int, default=3)
  parser.add_argument("--test-top-k", type=int, default=3)
  parser.add_argument("--constraint-weight", type=float, default=0.2)

  parser.add_argument("--no-fine-tune-roberta", type=bool, default=False)
  parser.add_argument("--use-softmax", type=bool, default=True)
  parser.add_argument("--debug", type=bool, default=False)
  parser.add_argument("--use-last-hidden-state", action="store_true")
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()
  print(args)

  # name = f"training_perc_{args.training_data_percentage}_seed_{args.seed}_clutrr"
  model_path = "/home/steinad/common-data/aadityanaik/scallop_models/clutrr/training_perc_100_seed_1234_clutrr_"

  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: device = torch.device("cpu")
  else: device = torch.device("cpu")

  # Setting up data and model directories
  data_root = "/home/steinad/common-data/unesy/"
  # model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/clutrr"))

  # Load the dataset
  (train_loader, test_loader) = clutrr_loader(data_root)

  # Train
  trainer = Trainer(train_loader, test_loader, device, model_path, args.model_name, args.learning_rate, num_mlp_layers=args.num_mlp_layers, debug=args.debug, provenance=args.provenance, train_top_k=args.train_top_k, test_top_k=args.test_top_k, use_softmax=args.use_softmax, no_fine_tune_roberta=args.no_fine_tune_roberta, load_model=args.load_model)
  # trainer.train(args.n_epochs)
  trainer.test_epoch(0)
