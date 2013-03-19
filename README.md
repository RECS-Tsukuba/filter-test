# 線形フィルタのテストツール

## 概要
画像に線形フィルタを適用し、表示します。表示される2枚の画像のうち、左が元の画像、右がフィルタ適用後の画像です。線形フィルタのオペレータはCSV形式で記述します。

## 必要環境
* OpenCV

## コンパイル時の必要環境
* C++11準拠のC++コンパイラ
* CMake
* OpenCVライブラリ

## コンパイル
### プロジェクトファイル、Makefileの生成
以下のコマンドを実行します。
> cmake .

WindowsならばMicrosoft Visual Studioのプロジェクトファイルを生成します。UNIX system likeな環境ならばMakefileを生成します。

### バイナリの生成 (Windows (未検証))
生成されたプロジェクトファイルをVisual Studioで開き、コンパイルします。

### バイナリの生成 (UNIX system likeな環境)
以下のコマンドを実行することで、バイナリが生成されます。 
> make

## 使い方
> linear\_filter image\_file filter\_csv

### image\_file
入力画像のファイル名。
### filter\_csv
フィルタのオペレータを記述したCSVファイル。整数だけでなく小数も扱うことができます。行の最後に絶対にカンマを入れないでください。
以下に例を示します。

ラプラシアン(3x3) 
<pre>
0,1,0 
1,-4,1 
0,1,0 
</pre>
 
エンボス(5x5)
<pre>
1,1,1,1,0
1,1,1,0,-1
1,1,0,-1,-1
1,0,-1,-1,-1
0,-1,-,1-,-1
</pre>
 
ガウシアン(3x3)
<pre>
0.0625,0.125,0.0625
0.125,0.25,0.125
0.0625,0.125,0.0625
</pre>


