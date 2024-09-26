# BiClass-Definition-Generator
BiClass-Definition-Generatorは，Zero-Shotでテキストを二値分類するための定義文を学習データから生成する手法をツール化したものです．

git cloneでソースコードをローカル環境にダウンロードし，設定ファイルの書き換えと自前のデータを準備することで，誰でも簡単に使用することが可能です．

なお，このソースコードでは，LLMsとしてOpenAIのモデルをAPI経由で使用しているため，ソースコードの実行にはOpenAIの登録およびAPI使用量が発生します．

GemmaやLlamaなどの他のLLMsを使用したい場合には，それらのモデルを使用するようにgenerate_text関数を自作することで，他のLLMsによる提案手法の実行が可能です．

## Installation
requires:
- openai==1.45.1
- openpyxl==3.1.5

opneaiはOpneAIのモデルをAPI経由で使用するのに使用します．

openpyxlはpandasで.xlsxのファイルを読み込むのに使用します．

## Quick Start
### repositoryのファイル一式をローカル環境にダウンロードする

#### git cloneによる方法
git clone https://github.com/k-takano0423/BiClass-Definition-Generator.git

#### zipファイルをダウンロードする方法
codeからDownload ZIPでzipファイルをダウンロードし，unzip BiClass-Definition-Generator-main.zipで解凍を行う

### 必要なlibraryのinstall
#### pip 経由
pip install openai==1.45.1
pip install openpyxl==3.1.5

### API Keyの設定
config_fileのディレクトリ内にあるconfig.jsonの「OPENAI_API_KEY」に，OpenAIから取得したAPI Keyを書き込む．

OpenAIのAPIに関しては，[こちら](https://platform.openai.com/docs/api-reference/introduction)を参考にしてください．

### BiClass Definition Generatorの実行
#### Auto BiClass Definition Generatorの実行
python3 ./src/ABCD_Gen.py
#### BiClass Definition Generator(Human In The Loop Mode)の実行
python3 ./src/BCD_Gen_HITL_Mode.py

## Description
#### 公開コードに関して
2種類のBiClass-Definition-Generatorを公開しております．

#### Auto BiClass Definition Generator(ABCD-Gen)
人手の入力が不要な全自動モードです．

#### Human In The Loop BiClass Definition Generator(BCD-Gen)
定義文の更新に使用する誤分類したテキストを人手で選択し，定義文も適宜修正が可能です．

適切なフィードバックを与えることにより，全自動よりもよりよい定義文の生成が期待できます．

### データに関して
Sample Dataとして，[内閣府が公開している景気ウォッチャー調査](https://www5.cao.go.jp/keizai3/watcher/watcher_menu.html)のテキストデータをdataというディレクトリにtrain.xlsx，test.xlsxとして配置しております．

必要なカラムは，「text」，「label」，「importance」の3つである．

「label」は1と0を割り当てる．(抽出対象のテキストを1とする)

「importance」は，各行のテキストの重要度で，ランダム抽出時に重み付けされて選択されることになります．最終的に合計が1となるように正規化されるため，0より大きい値を入力してください．(すべて1であれば等weightになります．)

新たにフォルダを作成して，そこに同じ形式でtrain.xlsx，test.xlsxを配置し，次の設定ファイルでpassを設定することで，ご自身で準備したデータに対して，ツールを適用することが可能です．

### 設定ファイルに関して

config_file内にあるconfig.jsonで様々な設定が可能です．

#### "OPENAI_API_KEY": ""
OpenAIのAPI Keyを設定してください．

#### "EXPERIMENT_NAME": "exp1",
実験に名前を付けることが可能です．

#### "DATA_PATH": "./data",
読み込むデータの配置先を指定します．

ご自身で準備したデータを配置したディレクトリを指定してください．

#### "OUTPUT_PATH": "./outputs",
生成した定義文やtestやvalidationの分類結果を出力するディレクトリを指定してください．

#### "MODEL_NAME": "gpt-4o-mini-2024-07-18",
OnenAIの使用するモデルを選択します．

#### "LANGUAGE": "ja",
使用するテキストの言語を設定します．

こちらで指定した言語の「prompt.json」を読み込んで入力するためのpromptを作成します．(多言語のprompt.jsonを共有いただければ追加させていただきます！)

#### "N_SPLIT": 4,
train.xlsxをラベルの割合が均一となるように，N_SPLITに分割します．

training data : validation data = (N_SPLIT - 1) : 1となるようにtrain_dfとvalid_dfにデータが分割されます．

#### "SEED_DEFINITION_STATEMENT": null,
一番最初の定義文の自動生成をせずに，人手で与える場合は，こちらに入力しておくことが可能です．(その場合，定義文の更新回数が+1されます．)

#### "SEED_GEN_TRUE_DATA_NUM": 10,
一番最初の定義文の自動生成に使用する正例のデータ数を指定します．

#### "SEED_GEN_FALSE_DATA_NUM": 10,
一番最初の定義文の自動生成に使用する負例のデータ数を指定します．

#### "MAX_SAMPLING_FALSE_NEGA_DATA_NUM": 2,
定義文更新時に使用するfalse negativeのデータ数を指定します．

#### "MAX_SAMPLING_FALSE_POSI_DATA_NUM": 2
定義文更新時に使用するfalse positiveのデータ数を指定します．


## Citation

```
@misc{BCD-Gen,
  title={BiClass Definition Generator},
  url={https://github.com/k-takano0423/BiClass-Definition-Generator},
  author={
    Kaito Takano and
    Kei Nakagawa and
    Yugo Fujimoto
  },
  year={2024},
}
```

## License

This code is licensed under the [Apache 2.0 LICENSE](LICENSE-2.0.txt)
