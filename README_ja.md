# MLOps-Basics

このシリーズの目標は、モデル構築、モニタリング、コンフィグレーション、テスト、パッケージング、デプロイメント、CI/CD など、MLOps の基本的なことを理解すること。

![pl](images/summary.png)

## 前提

以下の OCI サービスを活用します:

- OCI DataScience
  - Notebook セッションは、2OCPU, 32GB(メモリ), 100GB(ストレージ)以上であることが望ましい
- Oracle Functions
- API Gateway
- OCI Logging
- Service Connector Hub
- Search Service with OpenSearch

## toc

<!-- @import "[TOC]" {cmd="toc" depthFrom=3 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Week 0: Project Setup](#week-0-project-setup)
- [Week 1: Model monitoring - Weights and Biases](#week-1-model-monitoring---weights-and-biases)
- [Week 2: Configurations - Hydra](#week-2-configurations---hydra)
- [Week 3: Data Version Control - DVC](#week-3-data-version-control---dvc)
- [Week 4: Model Packaging - ONNX](#week-4-model-packaging---onnx)
- [Week 5: Model Packaging - Docker](#week-5-model-packaging---docker)
- [Week 6: CI/CD - GitHub Actions](#week-6-cicd---github-actions)
- [Week 7: Container Registry - AWS ECR](#week-7-container-registry---aws-ecr)
- [Week 8: Serverless Deployment - AWS Lambda](#week-8-serverless-deployment---aws-lambda)
- [Week 9: Prediction Monitoring - Kibana](#week-9-prediction-monitoring---kibana)

<!-- /code_chunk_output -->

## Weeks

### Week 0: Project Setup

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

参考: [https://www.ravirajag.dev/blog/mlops-project-setup-part1](https://www.ravirajag.dev/blog/mlops-project-setup-part1)

今週のスコープ:

- どのようにデータを取得するか？
- どのようにデータを処理するか？
- どのようにデータロードを定義するか？
- どのようにモデルを宣言するか？
- どのようにモデルを学習するか？
- どのように推論を行うか？

![pl](images/pl.jpeg)

技術スタック:

- [Huggingface Datasets](https://github.com/huggingface/datasets)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/)

### Week 1: Model monitoring - Weights and Biases

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

参考: [https://www.ravirajag.dev/blog/mlops-wandb-integration](https://www.ravirajag.dev/blog/mlops-wandb-integration)

ハイパーパラメーターを調整する、さまざまなモデルを試して性能をテストする、モデルと入力データとの関連性を確認するなど、すべての実験を追跡することは、より良いモデルを開発するのに役立つ。

今週のスコープ:

- W&B で基本的なロギングを設定する方法
- W&B でメトリクスを計算し、ログを取る方法
- W&B でプロットを追加する方法
- W&B にデータサンプルを追加する方法

![wannb](images/wandb.png)

技術スタック:

- [Weights and Biases](https://wandb.ai/site)
- [torchmetrics](https://torchmetrics.readthedocs.io/)

参考資料:

- [Tutorial on Pytorch Lightning + Weights & Bias](https://www.youtube.com/watch?v=hUXQm46TAKc)

- [WandB Documentation](https://docs.wandb.ai/)

### Week 2: Configurations - Hydra

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

参考: [https://www.ravirajag.dev/blog/mlops-hydra-config](https://www.ravirajag.dev/blog/mlops-hydra-config)

構成管理は、複雑なソフトウェアシステムを管理するために必要なものです。構成管理の欠如は、信頼性、アップタイム、およびシステムを拡張する能力に深刻な問題を引き起こす可能性があります。

今週のスコープ:

- Hydra の基本
- コンフィギュレーションのオーバーライド
- 複数のファイルに渡る設定の分割
- 変数補間
- 異なるパラメータの組み合わせでモデルを実行する方法

![hydra](images/hydra.png)

技術スタック:

- [Hydra](https://hydra.cc/)

参考資料:

- [Hydra Documentation](https://hydra.cc/docs/intro)

- [Simone Tutorial on Hydra](https://www.sscardapane.it/tutorials/hydra-tutorial/#executing-multiple-runs)

### Week 3: Data Version Control - DVC

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

参考: [https://www.ravirajag.dev/blog/mlops-dvc](https://www.ravirajag.dev/blog/mlops-dvc)

古典的なコードのバージョン管理システムは、大きなファイルを扱うように設計されていないため、クローンや履歴の保存は非現実的。どれが機械学習で非常によくあることなのか。

今週のスコープ:

- DVC の基本
- DVC の初期化
- リモートストレージの設定
- モデルをリモートストレージに保存する
- モデルのバージョン管理

![dvc](images/dvc.png)

技術スタック:

- [DVC](https://dvc.org/)

参考資料:

- [DVC Documentation](https://dvc.org/doc)

- [DVC Tutorial on Versioning data](https://www.youtube.com/watch?v=kLKBcPonMYw)

### Week 4: Model Packaging - ONNX

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

参考: [https://www.ravirajag.dev/blog/mlops-onn](https://www.ravirajag.dev/blog/mlops-onn)

なぜモデルのパッケージ化が必要なのか？モデルは機械学習フレームワーク（sklearn, tensorflow, pytorch など）を使って構築することができます。モバイル、ウェブ、ラズベリーパイといった異なる環境でモデルを展開したい、あるいは異なるフレームワークで実行したい（pytorch で学習し、tensorflow で推論する）かもしれません。
AI 開発者が様々なフレームワーク、ツール、ランタイム、コンパイラでモデルを使用できるようにするための共通のファイルフォーマットは、多くの助けになるでしょう。

これは、コミュニティプロジェクト`ONNX`によって実現される。

今週のスコープ:

- ONNX とは？
- 学習済みモデルを ONNX 形式に変換するには？
- ONNX Runtime とは？
- ONNX に変換されたモデルを ONNX Runtime で実行するには？
- 比較表

![ONNX](images/onnx.jpeg)

技術スタック:

- [ONNX](https://onnx.ai/)
- [ONNXRuntime](https://www.onnxruntime.ai/)

参考資料:

- [Abhishek Thakur tutorial on onnx model conversion](https://www.youtube.com/watch?v=7nutT3Aacyw)
- [Pytorch Lightning documentation on onnx conversion](https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html)
- [Huggingface Blog on ONNXRuntime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)
- [Piotr Blog on onnx conversion](https://tugot17.github.io/data-science-blog/onnx/tutorial/2020/09/21/Exporting-lightning-model-to-onnx.html)

### Week 5: Model Packaging - Docker

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>

参考: [https://www.ravirajag.dev/blog/mlops-docker](https://www.ravirajag.dev/blog/mlops-docker)

なぜパッケージングが必要なのでしょうか？私たちはアプリケーションを他の人と共有しなければならないかもしれません。

そのため、他の人がアプリケーションを実行するには、ホスト側で実行されたのと同じ環境を構築する必要があり、多くの手動設定とコンポーネントのインストールを意味します。

このような制限を解決するのが、コンテナという技術です。

アプリケーションをコンテナ化/パッケージ化することで、任意のクラウドプラットフォーム上でアプリケーションを実行し、マネージドサービスやオートスケール、信頼性などの利点を得ることができる。

アプリケーションのパッケージ化を行うための最も著名なツールは Docker🛳 です。

今週のスコープ:

- `FastAPI wrapper`
- `Basics of Docker`
- `Building Docker Container`
- `Docker Compose`

![Docker](images/docker_flow.png)

参考資料:

- [Analytics vidhya blog](https://www.analyticsvidhya.com/blog/2021/06/a-hands-on-guide-to-containerized-your-machine-learning-workflow-with-docker/)

### Week 6: CI/CD - GitHub Actions

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://www.ravirajag.dev/blog/mlops-github-actions)

CI/CD is a coding philosophy and set of practices with which you can continuously build, test, and deploy iterative code changes.

This iterative process helps reduce the chance that you develop new code based on a buggy or failed previous versions. With this method, you strive to have less human intervention or even no intervention at all, from the development of new code until its deployment.

In this post, I will be going through the following topics:

- Basics of GitHub Actions
- First GitHub Action
- Creating Google Service Account
- Giving access to Service account
- Configuring DVC to use Google Service account
- Configuring Github Action

![Docker](images/basic_flow.png)

References

- [Configuring service account](https://dvc.org/doc/user-guide/setup-google-drive-remote)

- [Github actions](https://docs.github.com/en/actions/quickstart)

### Week 7: Container Registry - AWS ECR

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://www.ravirajag.dev/blog/mlops-container-registry)

A container registry is a place to store container images. A container image is a file comprised of multiple layers which can execute applications in a single instance. Hosting all the images in one stored location allows users to commit, identify and pull images when needed.

Amazon Simple Storage Service (S3) is a storage for the internet. It is designed for large-capacity, low-cost storage provision across multiple geographical regions.

In this week, I will be going through the following topics:

- `Basics of S3`

- `Programmatic access to S3`

- `Configuring AWS S3 as remote storage in DVC`

- `Basics of ECR`

- `Configuring GitHub Actions to use S3, ECR`

![Docker](images/ecr_flow.png)

### Week 8: Serverless Deployment - AWS Lambda

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://www.ravirajag.dev/blog/mlops-serverless)

A serverless architecture is a way to build and run applications and services without having to manage infrastructure. The application still runs on servers, but all the server management is done by third party service (AWS). We no longer have to provision, scale, and maintain servers to run the applications. By using a serverless architecture, developers can focus on their core product instead of worrying about managing and operating servers or runtimes, either in the cloud or on-premises.

In this week, I will be going through the following topics:

- `Basics of Serverless`

- `Basics of AWS Lambda`

- `Triggering Lambda with API Gateway`

- `Deploying Container using Lambda`

- `Automating deployment to Lambda using Github Actions`

![Docker](images/lambda_flow.png)

### Week 9: Prediction Monitoring - Kibana

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>

Refer to the [Blog Post here](https://www.ravirajag.dev/blog/mlops-monitoring)

Monitoring systems can help give us confidence that our systems are running smoothly and, in the event of a system failure, can quickly provide appropriate context when diagnosing the root cause.

Things we want to monitor during and training and inference are different. During training we are concered about whether the loss is decreasing or not, whether the model is overfitting, etc.

But, during inference, We like to have confidence that our model is making correct predictions.

There are many reasons why a model can fail to make useful predictions:

- The underlying data distribution has shifted over time and the model has gone stale. i.e inference data characteristics is different from the data characteristics used to train the model.

- The inference data stream contains edge cases (not seen during model training). In this scenarios model might perform poorly or can lead to errors.

- The model was misconfigured in its production deployment. (Configuration issues are common)

In all of these scenarios, the model could still make a `successful` prediction from a service perspective, but the predictions will likely not be useful. Monitoring machine learning models can help us detect such scenarios and intervene (e.g. trigger a model retraining/deployment pipeline).

In this week, I will be going through the following topics:

- `Basics of Cloudwatch Logs`

- `Creating Elastic Search Cluster`

- `Configuring Cloudwatch Logs with Elastic Search`

- `Creating Index Patterns in Kibana`

- `Creating Kibana Visualisations`

- `Creating Kibana Dashboard`

![Docker](images/kibana_flow.png)
