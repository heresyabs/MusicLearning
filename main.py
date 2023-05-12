import os
import sys
import numpy as np

from magenta.music import midi_io
from magenta.protobuf import music_pb2
from magenta.music import melodies_lib
from magenta.music import melody_encoder_decoder
from magenta.music import constants
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

# コマンドライン引数から入力と出力のディレクトリを取得
args_input_dir = '/content/drive/MIDI学習用'
args_output_dir = '/content/drive/MIDI学習用'

# MIDIファイルを読み込み、NoteSequence形式に変換する関数
def convert_midi_dir_to_note_sequences(midi_dir):
    """MIDIファイルをNoteSequence形式に変換し、それらをリストとして返す"""
    file_list = os.listdir(midi_dir)
    note_sequences = []
    error_count = 0

    for file in file_list:
        if file.endswith('.mid') or file.endswith('.midi'):
            midi_file = os.path.join(midi_dir, file)
            try:
                sequence = midi_io.midi_file_to_note_sequence(midi_file)
                note_sequences.append(sequence)
            except (midi_io.MIDIConversionError, EOFError) as e:
                print('MIDIファイルの読み込みに失敗しました: %s' % midi_file)
                error_count += 1
                if error_count > 5:  # エラーが一定数を超えたら処理を停止
                    raise Exception("多数のMIDI読み込みエラーが発生しました")

    return note_sequences

# NoteSequence形式のデータをメロディへと変換し、学習データとテストデータに分割する関数
def convert_sequences_to_melodies_and_split_train_test(sequences, train_ratio=0.9):
    """NoteSequence形式をメロディに変換し、それらを学習データとテストデータに分割する"""
    steps_per_quarter = constants.DEFAULT_STEPS_PER_QUARTER  # 通常は4、必要に応じて変更

    melodies = []
    for ns in sequences:
        melody = melodies_lib.from_note_sequence(ns, steps_per_quarter)
        melodies.append(melody)

    melody_seqs = [melody_encoder_decoder.encode(melody) for melody in melodies]

    return train_test_split(melody_seqs, train_size=train_ratio, random_state=42)

# Kerasを使用してLSTMモデルを訓練する関数
def create_model(input_shape, output_dim, units=64):
    """メロディ生成用のLSTMモデルを作成する"""
    model = Sequential([
        LSTM(units, input_shape=input_shape,
        return_sequences=True),
        LSTM(units),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model

# メロディのリストからデータセットを作成する関数
def create_dataset(melodies, sequence_length):
    """メロディのリストからデータセットを作成する"""
    x = []
    y = []

    for melody in melodies:
        for i in range(len(melody) - sequence_length):
            x.append(melody[i:i+sequence_length])
            y.append(melody[i+sequence_length])

    x = np.array(x) / 127.0  # MIDIノート値（0-127）を正規化
    y = np.array(y) / 127.0

    return x, y

# モデルを指定したパスに保存する関数
def save_model(model, model_path):
    """モデルを指定したパスに保存する"""
    model.save(model_path)

# 指定したパスからモデルを読み込む関数
def load_model(model_path):
    """指定したパスからモデルを読み込む"""
    return keras.models.load_model(model_path)

# MIDIディレクトリからNoteSequenceを生成
midi_dir = args_input_dir
sequences = convert_midi_dir_to_note_sequences(midi_dir)
train_data, test_data = convert_sequences_to_melodies_and_split_train_test(sequences)

# 入力と出力の次元を設定。各シーケンスは100の時間ステップと38の可能なノートを持つと仮定
input_shape = (100, 38)
output_dim = 38  # データセット内のユニークなノートの数
units = 64  # LSTMのユニット数
model = create_model(input_shape, output_dim, units)

sequence_length = 100  # 予測に使用するノートシーケンスの長さ
x_train, y_train = create_dataset(train_data, sequence_length)
x_test, y_test = create_dataset(test_data, sequence_length)

epochs = 10  # 学習エポック数
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)

# モデルを保存
model_path = os.path.join(args_output_dir, 'model.h5')
save_model(model, model_path)

# 生成開始のためのプライマメロディ
primer = np.zeros((1, 100, 38))  # 前述と同じ形状を仮定
primer[0, 0, 60 % 38] = 1  # C4から開始

generated = model.predict(primer)

# ここで'generated'は各時間ステップでの可能なノート上の確率分布のシーケンスです。
# これを実際のノートに変換するには、各時間ステップで最も可能性の高いノートを選択するか、分布からサンプリングすることが考えられます。
