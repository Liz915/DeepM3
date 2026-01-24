echo "🚀 Starting DeepM3 Mixed Traffic (Alternating: Cold Start <-> Hot Cache)..."
echo "📊 Check Grafana at http://localhost:3000 (Time range: Last 5 minutes)"

# ==========================================
# Payload 定义
# ==========================================
# 1. 固定用户 (用于命中 System 1 / Cache)
FIXED_PAYLOAD='{"user_id":"vip_fixed_user","recent_items":[1,2],"recent_times":[0.1,0.2]}'

while true; do
  # ==========================================
  # A. 发送随机新用户
  # 预期: Cache Miss -> Router -> System 2 (~800ms)
  # ==========================================
  RANDOM_USER="user_$(date +%s)_$RANDOM"
  # 注意：这里不加 X-Demo-Mode Header，让系统自动路由
  curl -s -o /dev/null -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"$RANDOM_USER\", \"recent_items\":[1,2], \"recent_times\":[0.1, 0.2]}"

  # ==========================================
  # B. 发送固定老用户
  # 预期: Cache Hit -> System 1 (<2ms)
  # ==========================================
  curl -s -o /dev/null -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d "$FIXED_PAYLOAD"

  # ==========================================
  # C. 进度条与频率控制
  # ==========================================
  echo -n "."
  # 0.2秒间隔，保证 QPS 不会太低，同时给 Grafana 足够的数据点
  sleep 0.2 
done