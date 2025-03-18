import pandas as pd

# 定義各球隊的打擊成績網址
urls = [
    'https://baseballdata.jp/2021/1/ctop.html',
    'https://baseballdata.jp/2021/2/ctop.html',
    'https://baseballdata.jp/2021/3/ctop.html',
    'https://baseballdata.jp/2021/4/ctop.html',
    'https://baseballdata.jp/2021/5/ctop.html',
    'https://baseballdata.jp/2021/6/ctop.html',
    'https://baseballdata.jp/2021/7/ctop.html',
    'https://baseballdata.jp/2021/8/ctop.html',
    'https://baseballdata.jp/2021/9/ctop.html',
    'https://baseballdata.jp/2021/376/ctop.html',
    'https://baseballdata.jp/2021/11/ctop.html',
    'https://baseballdata.jp/2021/12/ctop.html'
]

# 初始化一個空的DataFrame來存儲所有球員數據
all_players = pd.DataFrame()

# 逐個抓取每個網址的數據
for url in urls:
    # 使用Pandas的read_html函式抓取表格數據
    tables = pd.read_html(url)
    # 假設所需的表格是第一個表格
    df = tables[0]
    # 將抓取的數據添加到總的DataFrame中
    all_players = pd.concat([all_players, df], ignore_index=True)
# 去除欄位名稱的空格，確保能正確存取欄位
all_players.columns = all_players.columns.str.strip()
# 將打席數轉換為數值類型，並將非數值的條目轉換為NaN
all_players['打 席 数'] = pd.to_numeric(all_players['打 席 数'], errors='coerce')

# 篩選打席數大於50的球員
filtered_players = all_players[all_players['打 席 数'] > 50]
# 使用 drop() 移除不必要的欄位，例如 '本塁打' 和 '出塁率'
columns_to_drop = ['一 軍','調 子','最 近 5 試 合','得 点 圏 打 数','得 点 圏 安 打','得 点 圏 打 率','U C 率','U C 本 塁 打','盗 塁 成 功 率','企 犠 打','犠 打','犠 打 成 功 率','代 打 数','代 打 安 打','代 打 率']  # 根據需求修改
filtered_players = filtered_players.drop(columns=columns_to_drop, errors='ignore')
# 將篩選後的數據導出為CSV檔案
filtered_players.to_csv('2021_qualified_hitters.csv', index=False, encoding='utf-8-sig')

