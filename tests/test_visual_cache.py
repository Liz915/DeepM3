import sys
import os

#  src
sys.path.append(os.getcwd())

from src.perception.visual_cache import VisualSemanticCache

def test_cache_logic():
    print(" Starting Visual Cache Unit Test...")
    
    # 
    cache = VisualSemanticCache(capacity=10)
    
    # 
    img_a = "http://example.com/shoe.jpg"
    img_b = "http://example.com/error_log.png"
    
    # 1.  ( Miss)
    print("\n[Step 1] Requesting Image A (First time)...")
    res1 = cache.get_analysis(img_a)
    
    #  key  semantic_tags
    print(f"Result Tags: {res1.get('semantic_tags', 'Key Not Found')}")
    
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    
    # 
    assert stats['visual_cache_misses'] == 1
    assert stats['visual_cache_hits'] == 0

    # 2.  ( Hit)
    print("\n[Step 2] Requesting Image A (Second time)...")
    res2 = cache.get_analysis(img_a)
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    
    assert stats['visual_cache_misses'] == 1 # Miss 
    assert stats['visual_cache_hits'] == 1   # Hit  +1
    
    # 3.  ( Miss)
    print("\n[Step 3] Requesting Image B...")
    res3 = cache.get_analysis(img_b)
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    assert stats['visual_cache_misses'] == 2

    print("\n Visual Cache Logic Passed!")

if __name__ == "__main__":
    test_cache_logic()