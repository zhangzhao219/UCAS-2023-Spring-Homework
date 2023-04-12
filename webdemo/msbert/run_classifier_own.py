import argparse
import shutil
from src.finetune_eval_model import BertCLSModel
from src.bert_model import BertConfig
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.common.parameter import Parameter
from convert_example import convert_text
from models.bert import BertForSequenceClassification


def torch_to_ms(model, torch_model,save_path):

    ms_param_dict = model.parameters_dict()

    update_torch_to_ms(torch_model, ms_param_dict, 'bert.embeddings.word_embeddings.weight','bert.bert_embedding_lookup.embedding_table')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.embeddings.token_type_embeddings.weight','bert.bert_embedding_postprocessor.embedding_table')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.embeddings.position_embeddings.weight','bert.bert_embedding_postprocessor.full_position_embeddings')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.embeddings.LayerNorm.weight','bert.bert_embedding_postprocessor.layernorm.gamma')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.embeddings.LayerNorm.bias','bert.bert_embedding_postprocessor.layernorm.beta')

    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.self.query.weight','bert.bert_encoder.layers.0.attention.attention.query_layer.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.self.query.bias','bert.bert_encoder.layers.0.attention.attention.query_layer.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.self.key.weight','bert.bert_encoder.layers.0.attention.attention.key_layer.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.self.key.bias','bert.bert_encoder.layers.0.attention.attention.key_layer.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.self.value.weight','bert.bert_encoder.layers.0.attention.attention.value_layer.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.self.value.bias','bert.bert_encoder.layers.0.attention.attention.value_layer.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.output.dense.weight','bert.bert_encoder.layers.0.attention.output.dense.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.output.dense.bias','bert.bert_encoder.layers.0.attention.output.dense.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.output.LayerNorm.weight','bert.bert_encoder.layers.0.attention.output.layernorm.gamma')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.attention.output.LayerNorm.bias','bert.bert_encoder.layers.0.attention.output.layernorm.beta')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.intermediate.dense.weight','bert.bert_encoder.layers.0.intermediate.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.intermediate.dense.bias','bert.bert_encoder.layers.0.intermediate.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.output.dense.weight','bert.bert_encoder.layers.0.output.dense.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.output.dense.bias','bert.bert_encoder.layers.0.output.dense.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.output.LayerNorm.weight','bert.bert_encoder.layers.0.output.layernorm.gamma')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.0.output.LayerNorm.bias','bert.bert_encoder.layers.0.output.layernorm.beta')

    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.self.query.weight','bert.bert_encoder.layers.1.attention.attention.query_layer.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.self.query.bias','bert.bert_encoder.layers.1.attention.attention.query_layer.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.self.key.weight','bert.bert_encoder.layers.1.attention.attention.key_layer.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.self.key.bias','bert.bert_encoder.layers.1.attention.attention.key_layer.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.self.value.weight','bert.bert_encoder.layers.1.attention.attention.value_layer.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.self.value.bias','bert.bert_encoder.layers.1.attention.attention.value_layer.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.output.dense.weight','bert.bert_encoder.layers.1.attention.output.dense.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.output.dense.bias','bert.bert_encoder.layers.1.attention.output.dense.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.output.LayerNorm.weight','bert.bert_encoder.layers.1.attention.output.layernorm.gamma')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.attention.output.LayerNorm.bias','bert.bert_encoder.layers.1.attention.output.layernorm.beta')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.intermediate.dense.weight','bert.bert_encoder.layers.1.intermediate.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.intermediate.dense.bias','bert.bert_encoder.layers.1.intermediate.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.output.dense.weight','bert.bert_encoder.layers.1.output.dense.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.output.dense.bias','bert.bert_encoder.layers.1.output.dense.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.output.LayerNorm.weight','bert.bert_encoder.layers.1.output.layernorm.gamma')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.encoder.layer.1.output.LayerNorm.bias','bert.bert_encoder.layers.1.output.layernorm.beta')

    update_torch_to_ms(torch_model, ms_param_dict, 'bert.pooler.dense.weight','bert.dense.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'bert.pooler.dense.bias','bert.dense.bias')
    update_torch_to_ms(torch_model, ms_param_dict, 'classifier.weight','dense_1.weight')
    update_torch_to_ms(torch_model, ms_param_dict, 'classifier.bias','dense_1.bias')
    
    save_checkpoint(model, save_path)

def update_torch_to_ms(torch_model, ms_param_dict, torch_key, ms_key):
    value = torch_model[torch_key].cpu().numpy()
    value = Parameter(Tensor(value), name=ms_key)
    ms_param_dict[ms_key].set_data(value)


def run_classifier():

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    model = BertCLSModel(bert_net_cfg, False, len(labels))

    torch_model = BertForSequenceClassification.from_pretrained(args.path)

    if args.convert:
        shutil.copyfile(args.path+"pytorch_model.bin",args.path+"pytorch_model.ckpt")
        torch_to_ms(model, torch_model.state_dict(),finetune_ckpt)

    bert_net_cfg.batch_size = 1
    net = BertCLSModel(bert_net_cfg, False, len(labels))
    net.set_train(False)
    param_dict = load_checkpoint(finetune_ckpt)
    load_param_into_net(net, param_dict)

    model = Model(net)

    input_features = convert_text(args.text, vocab_file, bert_net_cfg.seq_length)
    
    input_ids = Tensor(input_features.input_ids, mstype.int32)
    input_mask = Tensor(input_features.input_mask, mstype.int32)
    token_type_id = Tensor(input_features.segment_ids, mstype.int32)

    pooled_output, logits = model.predict(input_ids, input_mask, token_type_id)
    print(logits)
    print(pooled_output)
    print("预测类别：", labels[logits.asnumpy().argmax()])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Pytorch To MindSpore')
    parser.add_argument('--path', type=str, default="/app/text_dl/output_dir/IEMOCAP/bert-tiny/finetune/run/lr_0.0001_ep_5_bs_16_wp_0.1/best_model/")
    parser.add_argument('--text', type=str, default="hey zhangzhao")
    parser.add_argument('--convert', type=bool, default=True)

    args = parser.parse_args()

    bert_net_cfg = BertConfig(
        seq_length=512,
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        use_relative_positions=False,
        dtype=mstype.float32,
        compute_type=mstype.float32,
    )

    vocab_file =  args.path + "vocab.txt"
    finetune_ckpt = args.path + "ms_model.ckpt"

    labels = [
        {"label": "0", "label_desc": "news_story"},
        {"label": "1", "label_desc": "news_culture"},
        {"label": "2", "label_desc": "news_entertainment"},
        {"label": "3", "label_desc": "news_sports"}
    ]
    run_classifier()
