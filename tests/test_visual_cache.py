import sys
import os

# 确保能导入 src
sys.path.append(os.getcwd())

from src.perception.visual_cache import VisualSemanticCache

def test_cache_logic():
    print("🧪 Starting Visual Cache Unit Test...")
    
    # 初始化缓存
    cache = VisualSemanticCache(capacity=10)
    
    # 模拟图片输入
    img_a = "http://example.com/shoe.jpg"
    img_b = "http://example.com/error_log.png"
    
    # 1. 第一次请求 (应该 Miss)
    print("\n[Step 1] Requesting Image A (First time)...")
    res1 = cache.get_analysis(img_a)
    
    # 这里的 key 是 semantic_tags
    print(f"Result Tags: {res1.get('semantic_tags', 'Key Not Found')}")
    
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    
    # 断言
    assert stats['visual_cache_misses'] == 1
    assert stats['visual_cache_hits'] == 0

    # 2. 第二次请求相同图片 (应该 Hit)
    print("\n[Step 2] Requesting Image A (Second time)...")
    res2 = cache.get_analysis(img_a)
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    
    assert stats['visual_cache_misses'] == 1 # Miss 数不变
    assert stats['visual_cache_hits'] == 1   # Hit 数 +1
    
    # 3. 请求不同图片 (应该 Miss)
    print("\n[Step 3] Requesting Image B...")
    res3 = cache.get_analysis(img_b)
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    assert stats['visual_cache_misses'] == 2

    print("\n✅ Visual Cache Logic Passed!")

if __name__ == "__main__":
    test_cache_logic()