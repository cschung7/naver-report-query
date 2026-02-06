# Universal Chart System

범용 가격 차트 시스템 - 모든 마켓(KRX, USA, Japan, India, HongKong)에서 사용 가능

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│           chart_component.html (Standalone)              │
│     TradingView Lightweight Charts + 기술적 지표         │
└────────────────────────┬────────────────────────────────┘
                         │ JSON API
                         ▼
┌─────────────────────────────────────────────────────────┐
│              universal_chart.py                          │
│         ChartDataProvider (Python Class)                 │
└────────────────────────┬────────────────────────────────┘
                         │
    ┌────────┬───────────┼───────────┬────────┐
    ▼        ▼           ▼           ▼        ▼
  KRX      USA        Japan       India   HongKong
 CSV       CSV         CSV         CSV      CSV
```

## 파일 구조

```
shared/chart/
├── universal_chart.py      # Python 데이터 프로바이더
├── chart_component.html    # 독립 차트 컴포넌트
└── README.md              # 이 문서
```

## 사용 방법

### 1. Python API 통합 (Flask)

```python
from shared.chart.universal_chart import ChartDataProvider, create_chart_blueprint

# 방법 1: Blueprint 자동 생성
app.register_blueprint(create_chart_blueprint("USA", url_prefix="/usa/chart"))
# -> /usa/chart/ohlcv?symbol=AAPL&days=180

# 방법 2: 직접 사용
provider = ChartDataProvider(market_name="KRX")
data = provider.get_ohlcv("삼성전자", days=180)
```

### 2. HTML 컴포넌트 임베딩

```html
<!-- iframe으로 임베딩 -->
<iframe
    src="/shared/chart/chart_component.html?api=/usa/chart/ohlcv&symbol=AAPL&days=180&theme=dark"
    width="100%"
    height="550"
    frameborder="0">
</iframe>

<!-- 직접 include (권장) -->
<div id="chart-container"></div>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
    // chart_component.html의 스크립트 부분 복사
</script>
```

### 3. JavaScript API

```javascript
// 차트 컴포넌트가 로드된 후
UniversalChart.init();                    // 초기화
UniversalChart.setSymbol('TSLA');         // 심볼 변경
UniversalChart.setDays(365);              // 기간 변경
const data = UniversalChart.getData();    // 데이터 조회
```

## API 응답 포맷

```json
{
    "success": true,
    "symbol": "AAPL",
    "market": "USA",
    "total_records": 1000,
    "returned_records": 180,
    "date_range": {
        "start": "2025-07-01",
        "end": "2026-01-23"
    },
    "ohlcv": [
        {"time": "2025-07-01", "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.5, "volume": 50000000},
        ...
    ],
    "indicators": {
        "ma10": [{"time": "2025-07-01", "value": 150.5}, ...],
        "ma20": [...],
        "ma60": [...],
        "rsi": [...],
        "bb_upper": [...],
        "bb_middle": [...],
        "bb_lower": [...]
    }
}
```

## 지원 마켓

| Market | Path | 파일 수 |
|--------|------|---------|
| KRX | `/mnt/nas/AutoGluon/AutoML_Krx/KRXNOTTRAINED` | 2,923 |
| USA | `/mnt/nas/AutoGluon/AutoML_Usa/USANOTTRAINED` | 4,024 |
| Japan | `/mnt/nas/AutoGluon/AutoML_Japan/JAPANNOTTRAINED` | 2,611 |
| India | `/mnt/nas/AutoGluon/AutoML_India/INDIANOTTRAINED` | 2,049 |
| HongKong | `/mnt/nas/AutoGluon/AutoML_Hongkong/HONGKONGNOTTRAINED` | 2,501 |

## 기술적 지표

- **MA (이동평균)**: 10일, 20일, 60일 - 점선으로 표시
- **RSI**: 14일 기준 상대강도지수
- **Bollinger Bands**: 20일 기준, ±2 표준편차

## 커스텀 마켓 추가

```python
# 새 마켓 설정 추가
ChartDataProvider.MARKET_CONFIGS["CRYPTO"] = {
    "path": "/path/to/crypto/data",
    "file_pattern": "{symbol}.csv",
    "date_column": None,
    "encoding": "utf-8"
}

# 또는 직접 경로 지정
provider = ChartDataProvider(data_path="/custom/path")
```

## 요구사항

- Python 3.8+
- pandas
- numpy
- Flask (선택사항, Blueprint 사용시)
