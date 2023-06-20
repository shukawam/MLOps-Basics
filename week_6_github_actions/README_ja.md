**Note: ライブラリを探索し、その使い方を学ぶことが目的です。SOTA モデルを作るためではありません**。

## Requirements:

このプロジェクトは Python3.8 を使用しています

以下のコマンドで仮想 env を作成します：

```bash
conda create --name project-setup python=3.8
conda activate project-setup
```

必要なものをインストールします：

```bash
pip install -r requirements.txt
```

※Fork 元と一部依存ライブラリのバージョンを変更しています。（参考: [https://github.com/graviraja/MLOps-Basics/issues/31](https://github.com/graviraja/MLOps-Basics/issues/31)）

```diff
- datasets==1.6.2
+ datasets==2.10.1
- transformers==4.5.1
+ transformers==4.27.3
```

## Running

### Training

要件をインストールした後、モデルをトレーニングするために、単純に実行します：

```bash
python train.py
```

WandB 側で Project(MLOps Basics) の作成が必要でした。加えて、`train.py` にも entity を自身のユーザー名へと修正が必要です。

```diff
- wandb_logger = WandbLogger(project="MLOps Basics", entity="raviraja")
+ wandb_logger = WandbLogger(project="MLOps Basics", entity="shukawam")
```

### Monitoring

ログの最後にトレーニングが完了すると、次のように表示されます：

```bash
wandb: Synced 5 W&B file(s), 4 media file(s), 3 artifact file(s) and 0 other file(s)
wandb:
wandb: Synced proud-mountain-77: https://wandb.ai/raviraja/MLOps%20Basics/runs/3vp1twdc
```

リンクをクリックすると、すべてのプロットを含む wandb ダッシュボードが表示されます。

Follow the link to see the wandb dashboard which contains all the plots.

### Versioning data

Refer to the blog: [DVC Configuration](https://www.ravirajag.dev/blog/mlops-dvc)

### Exporting model to ONNX

モデルの学習が完了したら、以下のコマンドでモデルを変換します：

```bash
python convert_model_to_onnx.py
```

※onnx モジュールが存在しなくてエラー発生

```bash
# ... omit
ModuleNotFoundError: No module named 'onnx'
# ... omit
```

個別にインストールする

```bash
pip install onnx
```

ok. (`python convert_model_to_onnx.py`の実行後)

```bash
$ ls models/
best-checkpoint.ckpt  model.onnx
```

### Inference

#### Inference using standard pytorch

トレーニング後、コード内のモデルチェックポイントパスを更新して実行する

```bash
python inference.py
```

#### Inference using ONNX Runtime

```bash
python inference_onnx.py
```

### Google Service account

こちらの手順でサービスアカウントを作成します。: [Create service account](https://www.ravirajag.dev/blog/mlops-github-actions)

### Configuring dvc

```bash
dvc init
dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954
dvc remote modify storage gdrive_use_service_account true
dvc remote modify storage gdrive_service_account_json_file_path creds.json
```

`creds.json` は、サービスアカウント作成時に作成されるファイル

### Docker

Docker のインストール手順: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

コマンドを使用してイメージを構築します。

```shell
docker build -t inference:latest .
```

（任意）後ほど、OCIR へプッシュすることを考えるとコンテナイメージ名は以下のようにしておいた方が良いかもしれません。(region, path 等は任意)

```bash
docker build -t nrt.ocir.io/orasejapan/shukawam/inference:latest .
```

コマンドを使用してコンテナを実行します。

```shell
docker run -p 8000:8000 --name inference_container inference:latest
```

(or)

コマンドを使用してイメージのビルドとコンテナの実行をします。

```shell
docker-compose up
```

※OCI DataScience で実行している場合、Docker のランタイムが含まれていないため OCI DataScience に Docker をインストールするか Docker がセットアップされている環境にモデルをコピーすると良いです。Dockerfile も一部修正しています。

```diff
- RUN pip install -r requirements_prod.txt
+ RUN pip install -r requirements_inference.txt
```

また、使用している Base Image では `datasets==2.10.1` が非対応だったためインストール可能な最新バージョン `datasets==2.4.0` へ修正しています。

```diff
- datasets==2.10.1
+ datasets==2.4.0
```

### Running notebooks

ノートブックの実行に[Jupyter lab](https://jupyter.org/install)を使用しています。

virtualenv を使っているので、`jupyter lab`というコマンドを実行すると、virtualenv が使われる場合と使われない場合があります。

virutalenv を確実に使用するためには、`jupyter lab`を実行する前に以下のコマンドを実行してください。

```bash
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```

