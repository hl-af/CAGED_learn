import json
from transformers import BertTokenizer

import tokenizer_utils


class TripleProcessor:
    def __init__(self, tokenizer, input_path, output_path, max_length=512):
        self.ent2id = dict()
        self.rel2id = dict()
        self.tokenizer = tokenizer
        self.input_path = input_path
        self.output_path = output_path
        self.max_length = max_length
        self.ent2id, self.rel2id = self.build_data_train()

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def get_add_ent_id(self, ent):
        if ent in self.ent2id:
            ent_id = self.ent2id[ent]
        else:
            ent_id = len(self.ent2id)
            self.ent2id[ent] = ent_id
            self.id2ent[ent_id] = ent

        return ent_id

    def read_entity_from_id(self, filename='./entity2id.txt'):
        entity2id = {}
        with open(filename, 'r') as f:
            for line in f:
                if len(line.strip().split()) > 1:
                    entity, entity_id = line.strip().split(
                    )[0].strip(), line.strip().split()[1].strip()
                    entity2id[entity] = int(entity_id)
        return entity2id

    def read_relation_from_id(self, filename='./relation2id.txt'):
        relation2id = {}
        with open(filename, 'r') as f:
            for line in f:
                if len(line.strip().split()) > 1:
                    relation, relation_id = line.strip().split(
                    )[0].strip(), line.strip().split()[1].strip()
                    relation2id[relation] = int(relation_id)

        # relation corresponding to self loop
        # relation2id['self_loop'] = len(relation2id)
        return relation2id

    def build_data_train(self):
        entity2id = self.read_entity_from_id(self.output_path + '/entity2id.txt')
        relation2id = self.read_relation_from_id(self.output_path + '/relation2id.txt')
        return entity2id, relation2id

    def get_add_rel_id(self, rel):
        if rel in self.rel2id:
            rel_id = self.rel2id[rel]
        else:
            rel_id = len(self.rel2id)
            self.rel2id[rel] = rel_id
            self.id2rel[rel_id] = rel
        return rel_id

    def read_triples(self):
        print('Read begin!')
        triples = []
        for file in ["train", "valid", "test"]:
            with open(self.input_path + '/' + file + ".txt", "r") as f:
                for line in f.readlines():
                    try:
                        head, rel, tail = line.strip().split("\t")
                        triples.append((self.ent2id[head], self.rel2id[rel], self.ent2id[tail]))
                    except Exception as e:
                        print(f"Error reading line: {line.strip()}. Error: {e}")
        return triples

    def pad_list_to_length_8(self,input_list):
        target_length = 8
        current_length = len(input_list)

        # 如果当前长度小于目标长度
        if current_length < target_length:
            # 计算需要填充的0的数量
            padding_length = target_length - current_length
            # 填充0
            input_list.extend([0] * padding_length)

        return input_list

    def process_and_save_triples(self):
        triples = self.read_triples()
        tokenized_triples = {}
        print('triples长度:', len(triples))
        triples_len = len(triples)
        i = 0
        for head, rel, tail in triples:
            if i % 100 == 0:
                print('进度:{}%'.format(i / triples_len * 100))
            inputs = self.tokenizer(str(head), str(rel), str(tail), return_tensors='pt', padding=True,
                                    truncation=True, max_length=self.max_length)
            input_ids = inputs['input_ids'].tolist()[0]  # convert tensor to list
            input_ids = self.pad_list_to_length_8(input_ids)
            attention_masks = inputs['attention_mask'].tolist()[0]  # convert tensor to list
            attention_masks = self.pad_list_to_length_8(attention_masks)
            triple_key = tokenizer_utils.get_triple_key(head, rel, tail)
            tokenized_triples[triple_key] = {
                'input_ids': input_ids,
                'attention_masks': attention_masks
            }
            i = i + 1

        with open(self.output_path + '/tokenized_triples.json', 'w') as f:
            json.dump(tokenized_triples, f)
        print(f"Tokenized triples saved to {self.output_path}")



class TokenizedTripleReader:
    def __init__(self, input_file):
        self.input_file = input_file
        self.tokenized_triples = self.load_tokenized_triples()

    def load_tokenized_triples(self):
        with open(self.input_file, 'r') as f:
            tokenized_triples = json.load(f)
        return tokenized_triples

    def get_triple_data(self, head, rel, tail):
        triple_key = tokenizer_utils.get_triple_key(head, rel, tail)
        if triple_key in self.tokenized_triples:
            return self.tokenized_triples[triple_key]
        else:
            print(f"Triple ({head}, {rel}, {tail}) not found.")
            return None


if __name__ == "__main__":
    output_file = 'data/WN18RR'
    tokenizer = BertTokenizer.from_pretrained('checkpoints/bert-base-uncased')
    processor = TripleProcessor(tokenizer, input_path='data/WN18RR', output_path=output_file)
    processor.process_and_save_triples()
    print('save success')
    '''
    读取示例
    # Example of accessing a specific triple
    '''
    # reader = TokenizedTripleReader(input_file=output_file)
    # tokenized_triples = reader.load_tokenized_triples()
    # result = reader.get_triple_data("head_example", "rel_example", "tail_example")
    # if result:
    #     print("input_ids:", result['input_ids'])
    #     print("attention_masks:", result['attention_masks'])
