import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics

import matplotlib
from IPython.display import display, HTML


def sentence_color(sentence, attention, sentence_attention_value):
    def colorize(words, color_array):
        # words is a list of words
        # color_array is an array of numbers between 0 and 1 of length equal to words
        cmap = matplotlib.cm.get_cmap('binary')
        template = '<span class="barcode"; style="color: black; border-style: solid; border-color: {}">{}</span>'
        colored_string = ''
        for i, (word, color) in enumerate(zip(words, color_array)):
            color = matplotlib.colors.rgb2hex(cmap(color)[:3])
            colored_string += template.format(color, '&nbsp' + word + '&nbsp')
            if i % 10 == 0 and i != 0:
                colored_string += '<br>'
        return colored_string

    s = colorize(sentence, attention)
    s = f'<span class="barcode"; style="color: black;"> <b>Sent. attention value is {sentence_attention_value}</b></span><br>' + s

    # to display in ipython notebook
    return s


def attention_print(model, encoded_resume, length, resume):
    def model_word_forward(batch_data, lengths_word, hidden_state):
        batch_data = torch.nn.utils.rnn.pack_padded_sequence(batch_data, batch_first=True,
                                                             lengths=lengths_word)

        f_output, h_output = model.word_att_net.gru_word(batch_data,
                                                         hidden_state)  # feature output and hidden state output
        f_output_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(f_output, batch_first=True)

        u = torch.tanh_(model.word_att_net.attn_word(f_output_unpacked))
        a = F.softmax(model.word_att_net.context_word(u), dim=1)

        output = (a * f_output_unpacked).sum(1)

        return output, h_output, a

    def model_sent_forward(batch_data, hidden_state):
        f_output, h_output = model.sent_att_net.gru_sentence(batch_data,
                                                             hidden_state)  # feature output and hidden state output

        u = torch.tanh_(model.sent_att_net.attn_sentence(f_output))
        a = F.softmax(model.sent_att_net.context_sentence(u), dim=1)

        output = (a * f_output).sum(1)
        output = model.sent_att_net.fc(output)

        return output, h_output, a

    output_list = []
    word_att_list = []

    for i in range(encoded_resume.shape[1]):
        output, model.word_hidden_state, word_att = model_word_forward(encoded_resume[:, i],
                                                                       [length[i]],
                                                                       model.word_hidden_state)
        output_list.append(output)
        word_att_list.append(word_att)

    output = torch.cat(output_list, 0).unsqueeze(0)
    output, model.sent_hidden_state, sent_att = model_sent_forward(output, model.sent_hidden_state)

    sent_att = sent_att.flatten().tolist()

    assert len(sent_att) == len(resume)
    assert len(word_att_list) == len(resume)

    display_str = ''
    for i in range(len(sent_att)):
        display_str += '<p overflow-wrap="break-word";>' + sentence_color(resume[i],
                                                                          word_att_list[i].flatten().tolist(),
                                                                          sent_att[i]) + '</p>'

    display(HTML(display_str))


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


class WordAttention(nn.Module):
    def __init__(self, embed_size, hidden_size=50):
        """

        :param embed_size: embedding size
        :param hidden_size: size of hidden layers
        """
        super(WordAttention, self).__init__()

        self.attn_word = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.context_word = nn.Linear(2 * hidden_size, 1, bias=False)

        self.gru_word = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, data, lengths, hidden_state):
        """

        :param data: list of np.arrays or torch.Tensors
        :param hidden_state: of gru_word
        :return:
        """

        batch_data = torch.nn.utils.rnn.pack_padded_sequence(data, batch_first=True,
                                                             lengths=lengths)

        f_output, h_output = self.gru_word(batch_data, hidden_state)  # feature output and hidden state output

        f_output_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(f_output, batch_first=True)

        u = torch.tanh_(self.attn_word(f_output_unpacked))
        a = F.softmax(self.context_word(u), dim=1)

        output = (a * f_output_unpacked).sum(1)

        return output, h_output


class SentenceAttention(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentenceAttention, self).__init__()

        self.attn_sentence = nn.Linear(2 * sent_hidden_size, 2 * sent_hidden_size)
        self.context_sentence = nn.Linear(2 * sent_hidden_size, 1, bias=False)

        self.gru_sentence = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)

    def forward(self, batch_data, hidden_state):
        f_output, h_output = self.gru_sentence(batch_data, hidden_state)  # feature output and hidden state output

        u = torch.tanh_(self.attn_sentence(f_output))
        a = F.softmax(self.context_sentence(u), dim=1)

        output = (a * f_output).sum(1)
        output = self.fc(output)

        return output, h_output


class HAN(nn.Module):
    def __init__(self, word_embed_size, word_hidden_size, sent_hidden_size, batch_size, num_classes):
        super(HAN, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttention(word_embed_size, word_hidden_size)
        self.sent_att_net = SentenceAttention(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self):
        self.word_hidden_state = torch.zeros(2, self.batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, self.batch_size, self.sent_hidden_size)

    def forward(self, input_data, lengths_of_data):
        """
        input_data: [1, num_sentences, num_words, emb_word_size]
        lengths_of_data: list of lengths foe each sentence (without padding)
        """

        output_list = []

        for i in range(input_data.shape[1]):
            output, self.word_hidden_state = self.word_att_net(input_data[:, i],
                                                               [lengths_of_data[i]],
                                                               self.word_hidden_state)
            output_list.append(output)

        output = torch.cat(output_list, 0).unsqueeze(0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)

        return output


def train(model, train_dataloader, test_dataloader, num_epoch=15, test_interval=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)

    for epoch in range(num_epoch):
        acc_temp = 0
        loss_temp = 0
        for feature, lengths, label in train_dataloader:
            optimizer.zero_grad()
            model._init_hidden_state()

            predictions = model(feature, [int(i[0]) for i in lengths])

            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()

            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy"])
            acc_temp += training_metrics["accuracy"]
            loss_temp += loss

        acc_temp = acc_temp / len(train_dataloader)
        loss_temp = loss_temp / len(train_dataloader)

        print("(Train) Epoch: {}/{} Loss: {}, Accuracy: {}".format(
            epoch + 1,
            num_epoch,
            loss_temp, acc_temp))

        if epoch % test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_length, te_label in test_dataloader:
                with torch.no_grad():
                    model._init_hidden_state()
                    te_predictions = model(te_feature, [int(i[0]) for i in te_length])

                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())

            te_loss = sum(loss_ls) / len(loss_ls)
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])

            print("(Test)  Epoch: {}/{}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                num_epoch,
                te_loss, test_metrics["accuracy"]))

    return model


if __name__ == '__main__':
    from dataset import create_dataset

    (emb_matrix, word_to_ind, ind_to_word, ind_to_freq, vocabulary,
     train_all_dataset, test_all_dataset, train_loader, test_loader,
     dataframe) = create_dataset('UpdatedResumeDataSet.csv', 'emb_matrix.npy')

    HAN_model = HAN(emb_matrix.shape[1], word_hidden_size=50, sent_hidden_size=50,
                    batch_size=1, num_classes=len(dataframe['Category'].unique()))

    HAN_model = train(HAN_model, train_loader, test_loader)

    for i in range(0, len(train_all_dataset), 5):
        resume, length, label = train_all_dataset[i]
        row = dataframe.iloc[i, :]

        print('Resume label', label, ',', row['Category'])
        print('\n\n')
        for sent in row['Resume']:
            print(sent)
        print('\n\n')
        print('Predict', HAN_model(resume.unsqueeze(0), length))
        attention_print(HAN_model, resume.unsqueeze(0), length, row['Resume'])
        print('\n\n', '-' * 10)
