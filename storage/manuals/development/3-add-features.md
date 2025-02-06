# 新機能の追加

このドキュメントは、開発手法にイシュードリブン開発を採用して開発する手順を説明しています。

## イシュードリブン開発とは

新機能の開発は、必ずGitHub Issueと紐付けて行います。これにより:

- 開発の目的と要件が明確になる
- 他の開発者との作業の重複を防ぐ
- 進捗の追跡が容易になる

## イシューの作成

まずは開発目的を記載するイシューを作成します。

1. [Issues](https://github.com/connect0459/py_flux_tracer/issues) からイシューページに飛ぶ
2. 「New issue」をクリックして、タイトル、説明などを入力して「Submit new issue」で新しいイシューを作成

## ブランチの作成

### ブランチの命名規則

開発用のブランチは以下の形式で命名します:

```bash
develop/#<Issue番号>_<機能の説明>
```

以下に主に使用されるブランチのプレフィックスです。

| ブランチ名 | 役割 | 派生元 | マージ先 |
| :--- | :--- | :--- | :--- |
| main | 公開するものを置くブランチ |  |  |
| develop | 開発中のものを置くブランチ | main | main |
| release | 次にリリースするものを置くブランチ | develop | develop, main |
| feature-* | 新機能開発中に使うブランチ | develop | develop |
| hotfix-* | 公開中のもののバグ修正用ブランチ | main | develop, main |

`feature-*` や `hotfix-*` は、`*`に適当な名前をつける。

例:

- `develop/#2_implement_data_export` (データエクスポート機能の実装)
- `feature/#7_add_graph_visualization` (グラフ可視化機能の追加)
- `hotfix/#12_fix_memory_leaks` (メモリリークの修正)

### ブランチの作成手順

```bash
# mainブランチを最新化
git checkout main
git pull origin main

# 新機能用のブランチを作成
git checkout -b develop/#1_add_foo
```

## 開発作業

### コミットのガイドライン

- 論理的な単位で小さくコミットする
- コミットメッセージは明確に記述する
- Issue番号を含める

```bash
# 変更をステージング
git add .

# コミット
git commit -m "dev(#1): ログイン画面のUIを実装"
```

### 作業中のプッシュ

定期的にリモートリポジトリにプッシュすることで、作業内容のバックアップと共有を行います:

```bash
# 初回プッシュ
git push -u origin develop/#1_add_foo

# 以降のプッシュ
git push origin develop/#1_add_foo
```

## mainブランチへのマージ

### プルリクエストの作成

1. GitHubのリポジトリページで「Pull requests」を選択
2. 「New pull request」をクリック
3. 以下の情報を記入:
   - タイトル: 変更の概要
   - 説明:
     - 実装した機能の詳細
     - テスト結果
     - 関連するIssue(`#1`のように記述)
4. レビュアーを指定

### マージ

レビューが承認されたら、以下の手順でマージを実行:

1. 「Merge pull request」をクリック
2. 必要に応じてマージコミットメッセージを編集
3. マージを確定
4. 作業ブランチを削除(オプション)
