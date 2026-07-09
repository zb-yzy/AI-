import os
import time
import random
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pickle

# ======================= 【你只需要改这里！】=======================
TARGET_MOVIE_COUNT = 1000      # 想要多少部电影？建议 500 / 800 / 1000
COMMENTS_PER_MOVIE = 70        # 每部电影爬几条评论
# ==================================================================

# 固定配置
MOVIE_CSV = "豆瓣电影列表.csv"
COMMENT_CSV = "电影评论数据.csv"
POSTER_FOLDER = "电影海报"
os.makedirs(POSTER_FOLDER, exist_ok=True)

# 浏览器配置
chrome_opt = Options()
chrome_opt.add_argument("--window-size=1200,800")
chrome_opt.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(options=chrome_opt)
driver.implicitly_wait(5)

# 请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://movie.douban.com/"
}

def delay():
    time.sleep(1 + random.random())

# ============================================
# 1. 电影列表获取（支持从已有CSV恢复）
# ============================================
def get_movie_list(target_count):
    """如果已有CSV且数量足够则直接读取，否则重新爬取"""
    if os.path.exists(MOVIE_CSV):
        df_existing = pd.read_csv(MOVIE_CSV, encoding="utf-8-sig")
        if len(df_existing) >= target_count:
            print(f"✅ 发现已有文件 {MOVIE_CSV}，包含 {len(df_existing)} 部电影，达到目标 {target_count}，直接使用。")
            movies = df_existing.head(target_count).to_dict('records')
            return movies
        else:
            print(f"⚠️ 发现已有文件 {MOVIE_CSV}，但只有 {len(df_existing)} 部，不足 {target_count}，将重新爬取。")
    else:
        print(f"📁 未找到 {MOVIE_CSV}，开始爬取电影列表。")

    print(f"🚀 开始爬取电影列表，目标：{target_count} 部（通过API动态获取）")
    api_url = "https://movie.douban.com/j/search_subjects"
    tags = ['热门', '最新', '经典', '可播放', '豆瓣高分', '冷门佳片',
            '华语', '欧美', '韩国', '日本', '动作', '喜剧', '爱情',
            '科幻', '悬疑', '恐怖', '动画']
    all_movies = []
    start = 0

    while len(all_movies) < target_count:
        tag = random.choice(tags)
        params = {
            'type': 'movie',
            'tag': tag,
            'sort': 'recommend',
            'page_limit': 20,
            'page_start': start
        }
        try:
            resp = requests.get(api_url, headers=HEADERS, params=params, timeout=10)
            if resp.status_code != 200:
                print(f"请求失败，状态码: {resp.status_code}，切换标签...")
                start = 0
                continue

            data = resp.json()
            subjects = data.get('subjects', [])
            if not subjects:
                start = 0
                continue

            for movie in subjects:
                if len(all_movies) >= target_count:
                    break
                if not any(m.get('电影ID') == movie['id'] for m in all_movies):
                    all_movies.append({
                        '电影ID': movie['id'],
                        '电影名': movie['title'],
                        '海报URL': movie['cover']
                    })

            print(f"🎬 从标签『{tag}』获取 {len(subjects)} 部，累计: {len(all_movies)}/{target_count}")
            start += 20
            delay()
        except Exception as e:
            print(f"⚠️ API请求出错: {e}，稍后重试...")
            time.sleep(3)
            continue

    df = pd.DataFrame(all_movies[:target_count])
    df.to_csv(MOVIE_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 电影列表保存完成：{MOVIE_CSV}，共 {len(df)} 部电影")
    return all_movies[:target_count]

# ============================================
# 2. 下载海报
# ============================================
def download_poster(mid, name, img_url):
    try:
        safe_name = "".join([c for c in name if c not in '\\/:*?"<>|'])
        save_path = os.path.join(POSTER_FOLDER, f"{mid}_{safe_name}.jpg")
        resp = requests.get(img_url, headers=HEADERS, timeout=10)
        if resp.status_code != 200 or len(resp.content) < 2000:
            return ""
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path
    except:
        return ""

# ============================================
# 3. 爬取评论 + 海报（支持断点续爬、追加模式）
# ============================================
def crawl_all(movies):
    """自动跳过已爬电影，从断点继续，每爬一部立即保存"""
    # 加载已有评论，记录已爬电影ID
    existing_ids = set()
    if os.path.exists(COMMENT_CSV):
        existing_df = pd.read_csv(COMMENT_CSV, encoding="utf-8-sig")
        if not existing_df.empty:
            existing_ids = set(existing_df["电影ID"].astype(str))
            print(f"📂 发现已有评论文件，包含 {len(existing_ids)} 部电影的数据")
    else:
        existing_df = pd.DataFrame()
        print("📁 未找到已有评论文件，将新建")

    # 确定起始索引（自动找到第一个未爬的电影）
    start_idx = 0
    for i, m in enumerate(movies):
        if str(m["电影ID"]) not in existing_ids:
            start_idx = i
            break
    print(f"🔁 自动从第 {start_idx+1} 部电影开始（共跳过 {start_idx} 部）")

    # 如果已经全部爬完
    if start_idx >= len(movies):
        print("✅ 所有电影均已爬取过，无需继续。")
        driver.quit()
        return

    # 准备数据容器（已有 + 新增）
    all_data = existing_df.to_dict('records') if not existing_df.empty else []

    # 初始化浏览器并加载登录cookies
    driver.get("https://movie.douban.com/")
    time.sleep(2)
    try:
        if os.path.exists("douban_cookies.pkl"):
            cookies = pickle.load(open("douban_cookies.pkl", "rb"))
            for cookie in cookies:
                driver.add_cookie(cookie)
            driver.refresh()
            time.sleep(2)
            print("✅ 已加载豆瓣登录 Cookies，可查看完整评论")
        else:
            print("⚠️ 未找到 douban_cookies.pkl，将以未登录状态爬取（只能看到前20条）")
    except Exception as e:
        print(f"⚠️ Cookies 加载失败: {e}，继续尝试未登录爬取")

    total = len(movies)
    for idx in range(start_idx, total):
        info = movies[idx]
        mid = info["电影ID"]
        name = info["电影名"]
        poster_url = info["海报URL"]
        poster_path = download_poster(mid, name, poster_url)

        url = f"https://movie.douban.com/subject/{mid}/comments"
        print(f"[{idx+1}/{total}] 正在处理: {name} (ID: {mid})")

        # 防止重复（二次确认）
        if str(mid) in existing_ids:
            print(f"  ⏭️ 已存在 {mid} 的数据，跳过")
            continue

        # 访问评论页，带重试机制
        max_retry = 3
        page_ok = False
        for retry in range(max_retry):
            try:
                driver.get(url)
                time.sleep(3)

                # 检查是否被重定向到登录页
                if "login" in driver.current_url:
                    print("  ⚠️ 被重定向到登录页，尝试重新加载cookies...")
                    driver.get("https://movie.douban.com/")
                    if os.path.exists("douban_cookies.pkl"):
                        for cookie in pickle.load(open("douban_cookies.pkl", "rb")):
                            driver.add_cookie(cookie)
                        driver.refresh()
                        time.sleep(2)
                        driver.get(url)
                        time.sleep(3)
                    if "login" in driver.current_url:
                        print("  ❌ 仍然被跳转到登录页，跳过此电影")
                        break
                    else:
                        page_ok = True
                        break
                else:
                    page_ok = True
                    break
            except Exception as e:
                print(f"  ⚠️ 第{retry+1}次加载失败: {e}")
                time.sleep(5)
        if not page_ok:
            print(f"  ❌ 跳过 {name}，无法加载评论页")
            continue

        # 翻页收集评论（最多 COMMENTS_PER_MOVIE 条，最多10页）
        collected = []
        current_page = 1
        while len(collected) < COMMENTS_PER_MOVIE and current_page <= 10:
            try:
                # 等待评论出现
                for _ in range(10):
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    items = soup.find_all("div", class_="comment-item")
                    if items:
                        break
                    time.sleep(1)

                if not items:
                    print(f"    第{current_page}页无评论，停止翻页")
                    break

                for item in items:
                    text_tag = item.find("span", class_="short")
                    if not text_tag:
                        continue
                    content = text_tag.get_text(strip=True)
                    star = 0
                    star_tag = item.find("span", class_=lambda c: c and "rating" in c)
                    if star_tag:
                        cls = star_tag["class"][0]
                        num = cls.replace("allstar", "").replace("rating", "")
                        star = int(int(num) // 10)
                    collected.append({
                        "电影ID": mid,
                        "电影名": name,
                        "海报路径": poster_path,
                        "星级": star,
                        "评论": content
                    })
                    if len(collected) >= COMMENTS_PER_MOVIE:
                        break

                if len(collected) >= COMMENTS_PER_MOVIE:
                    break

                # 点击下一页
                try:
                    next_btn = driver.find_element(By.CSS_SELECTOR, "a.next")
                    driver.execute_script("arguments[0].scrollIntoView();", next_btn)
                    time.sleep(0.5)
                    next_btn.click()
                    time.sleep(2)
                    current_page += 1
                except:
                    print(f"    没有更多页面（当前共{len(collected)}条）")
                    break
            except Exception as e:
                print(f"    翻页出错: {e}")
                break

        # 追加到总数据并立即保存
        all_data.extend(collected[:COMMENTS_PER_MOVIE])
        print(f"  ✅ 获得 {len(collected[:COMMENTS_PER_MOVIE])} 条评论")
        pd.DataFrame(all_data).to_csv(COMMENT_CSV, index=False, encoding="utf-8-sig")

        # 每10部额外打印一次保存信息（实际每部都已保存）
        if (idx + 1) % 10 == 0:
            print(f"  💾 已保存进度到 {COMMENT_CSV}")

    driver.quit()
    print(f"\n🎉 全部完成！共爬取 {len(all_data)} 条评论，涉及 {len(set([d['电影ID'] for d in all_data]))} 部电影")

# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    # 1. 获取电影列表（自动判断是否已有足够数据）
    movie_list = get_movie_list(TARGET_MOVIE_COUNT)
    # 2. 爬取评论和海报（自动断点续爬，不覆盖已有数据）
    crawl_all(movie_list)