[必要なライブラリ]
numpy

scipy

scikit-learn

scikit-image

tqdm

av

timm

PySide6

OpenCV

PyTorch
v1.6以上
GPUを使用する場合は、対応したバージョンをインストールして下さい。

torchvision
V0.7.0以上

FrEIA Flows
https://github.com/vislearn/FrEIA/archive/cc5cf5ebee08f9bb762bab5a6535c11d19ccb026.zip
解凍後、フォルダ名を分かり易い名前に変更。
コマンドプロンプトからフォルダの場所を開く(例 cd c:\FrEIA)。
python setup.py developを実行。

必要があれば、他のライブラリもインストールして下さい。



[実行ファイル]
学習用実行ファイル
MAIN_TRAINER.py

検出用実行ファイル
MAIN_DETECTOR_MOV.py



[運用の流れ]
・高い精度で異常を検出する前提。
1.カメラは一定の位置に固定する。
2.対象物の位置、方向は一定にする（微妙なズレは問題ありません）。
3.照明を当て、環境が一定の明るさになるようにする事が非常に重要です。
4.学習する画像と、検出する画像を同じ絵図にする（ディテクションサイズ、ビデオサイズ、トリムを一定にする）。

・学習用の画像収集
1.MAIN_DETECTOR_MOV.pyを実行。
2.[DETECTION SIZE]、[VIDEO SIZE]を決定する。
3.[TAKE PICTURE]ボタンを押し、学習用の画像を保存するフォルダを選択。
4.検出精度を上げる為に、必要に応じてマウスを左クリックしながら、領域を選択してください（間違えた場合は、[STOP]ボタンを押した後、[CLEAR]ボタンを押して下さい）。
5.スペースキーを押して、画像を保存して下さい。
6.十分に画像を保存したら、[STOP]ボタンを押して、終了して下さい（画像は250枚程度有ると良いかと思います）。
7.実際に検出する際、同じ環境で実行出来る様、[SAVE]ボタンを押して、設定を保存して下さい。

・学習
1.MAIN_TRAINER.pyを実行。
2.もし、学習用の画像が連番でない場合は、[RENUMBER]を押して画像が保存してあるフォルダを選択し、ファイル名を連番にして下さい。
3.MAIN_DETECTOR_MOV.pyの[DETECTION SIZE]と、MAIN_TRAINER.py[TRAIN SIZE]を同じにして下さい。
4.[OPEN]ボタンを押し、学習用画像が保存されているフォルダを選択して下さい。
5.[START]ボタンを押して、学習をスタートして下さい（1～3時間程度で学習が終了すると思います）。

・検出の準備
1.1.MAIN_DETECTOR_MOV.pyを実行。
1.[LOAD]ボタンを押し、画像を収集した際の設定を読み込んで下さい。
2.[OPEN]ボタンを押し、学習データ、拡張子がptの最新ファイルを選択して下さい。
3.[KERNEL SIZE]を設定(異常検知の検出単位で、値が大きくなる程、小さい異常を検出しなくなる)。
4.[THRESHOLD FOR DETECTION]を設定。異常とする閾値。事前に、[CHECK]ボタンで良品の写真のフォルダーを選択し、推奨値を取得し設定（学習した際のIMAGE SIZEを選択）。
5.[THRESHOLD FOR AREA SIZE]を設定。異常として検出したエリアを、どの程度の大きさなら異常とするかの閾値。
6.[NON SIMILER THRESHOLD]を設定。学習した画像と全く違う画像を判定する際に、異常として判定する加算値（通常は設定変更不要）。
7.[SAVE]ボタンを押し、設定を保存する。

・検出
1.[LOAD]ボタンを押し、設定を読み込んで下さい。
2.[START]ボタンを押し、検出を開始します。
3.MAIN_LOOP()のDETETED_PICTURE = DETECTION(frameB)を実行後、変数のDETECT_COUNTERに不具合数が格納されます。
4.必要に応じて外部出力等の処理を、DETETED_PICTURE = DETECTION(frameB)直後に追記して下さい。
