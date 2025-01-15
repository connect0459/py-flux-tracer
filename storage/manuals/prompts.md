# プロンプト

## docstringのフォーマットの統一

```md
このdocstringを以下の要件に基づいてNumPy/SciPy Styleのフォーマットに書き変えてください。
- セクション（Parameters, Returns等）を示し、破線（-----）で区切る。その後にパラメータをまとめて記述。
- パラメータはname : typeの形式で記述。
- 説明は4スペースのインデントで記述。
- 戻り値の型はReturnsセクションで詳細に記述。
例↓
Parameters:
------
    hoge : str
        hogehoge...
    fuga : int
        fugafuga...

Returns:
------
    ...
```
