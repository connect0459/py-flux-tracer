# プロンプト

## docstringのフォーマットの統一

```md
このdocstringを以下の要件に基づいてNumPy/SciPy Styleのフォーマットに書き変えてください。
- セクション(Parameters, Returns等)を示し、破線(-----)で区切る。その後にパラメータをまとめて記述。
- パラメータはname: typeの形式で記述。
- 説明は4スペースのインデントで記述。
- 戻り値の型はReturnsセクションで詳細に記述。
例↓
Parameters
----------
    hoge: str
        hogehoge...
    fuga: int
        fugafuga...

Returns
----------
    ...
```

## 2

このメソッドのdocstringを、以下の要件に基づいて更新してください。

- 引数の横にコメントを付けることを禁じます。
- 引数にあってdocstringに説明がない場合は、docstringに説明を追加してください。
- カッコやコロンなどの記号は半角を使用してください。
- 引数にデフォルト値がある場合はデフォルト値に関する説明を加え、型注釈の後に",optional"を加えてください。
- docstringに英語と日本語が混ざっている場合は日本語に統一してください。
- publicなメソッドにはExamplesセクションを追加して使い方を説明してください。
- ParametersセクションとReturnsセクションのコンテンツは、インデントを4スペース分空けてください。Examplesは空けなくてよいです。例:

```txt
"""
...
        Parameters
        ----------
            output_dirpath: str | Path | None
                保存先のディレクトリを指定します。save_fig=Trueの場合は必須です。
...
```
