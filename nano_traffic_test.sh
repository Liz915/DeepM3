echo "🚀 Starting Logically Correct Traffic (L1/L2/L3)..."

FIXED_CHAOS_USER="user_chaos_fixed_007"
CHAOS_BODY='{"user_id":"user_chaos_fixed_007","recent_items":[10, 300, 5, 1000, 50], "recent_times":[0.1, 2.5, 3.1, 8.0, 15.2] }'
STABLE_BODY='{"user_id":"user_stable_loop","recent_items":[1,1,1,1,1], "recent_times":[0.1,0.1,0.1,0.1,0.1]}'

while true; do
    # 1) L3 Slow Path（必 miss）
    curl -s -o /dev/null -X POST http://localhost:8000/recommend \
        -H "Content-Type: application/json" \
        -d "$CHAOS_BODY"

    # 2) 等后台写缓存（0.6 就够）
    sleep 0.6

    # 3) L1 Cache Hit（三次稳定命中）
    for i in {1..3}; do
        curl -s -o /dev/null -X POST http://localhost:8000/recommend \
            -H "Content-Type: application/json" \
            -d "$CHAOS_BODY"
    done

    # 4) L2 Fast Path（提高 QPS + 让图更好看）
    for i in {1..5}; do
        curl -s -o /dev/null -X POST http://localhost:8000/recommend \
            -H "Content-Type: application/json" \
            -d "$STABLE_BODY"
    done

    echo -n "."
done