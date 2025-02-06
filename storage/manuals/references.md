# リファレンス

このドキュメントでは、`py_flux_tracer`パッケージで提供されている機能について説明します。

## データ前処理

- `EddyDataFiguresGenerator`: 渦相関法で得られたデータから図を作成するクラスです。
- `EddyDataPreprocessor`: 渦相関法で得られたデータの前処理を行うクラスです。
- `SpectrumCalculator`: スペクトル解析を行うためのクラスです。

## フットプリント

- `FluxFootprintAnalyzer`: フラックスフットプリントを計算し、可視化するためのクラスです。

## 車載濃度観測

- `CorrectingUtils`: 観測データに補正を行う処理をまとめたクラスです。
- `HotspotEmissionAnalyzer`: `MobileMeasurementAnalyzer`で計算された`HotspotData`をもとに、CH4排出量を解析するクラスです。
- `MobileMeasurementAnalyzer`: 車載濃度観測データを解析するためのクラスです。

## 月別データ

- `MonthlyConverter`: MonthlyシートをDataFrameに変換するためのクラスです。
- `MonthlyFiguresGenerator`: Monthlyシートから作成されたDataFrameから図を作成するクラスです。

## 伝達関数

FluxCalculator から出力されるFFTファイルをもとに伝達関数の係数を計算します。

- `FftFileReorganizer`: FFTファイルを整理するためのクラスです。
- `TransferFunctionCalculator`: 伝達関数を計算し、可視化するためのクラスです。
