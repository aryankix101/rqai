# Pipeline runner for multi-ticker backtesting

set -e 

# Configuration
DATA_START_DATE="1993-01-01"      
BACKTEST_START_DATE="2000-01-01"  
END_DATE="2025-12-31"
FORWARD_HORIZONS=(5 10 20 30 60)
RUN_STAGE1=false                  # Set to true to regenerate signal data, false to skip
RUN_STAGE2=false                  # Set to true to regenerate composite scores, false to skip

# Ticker list
TICKERS=(
AABA AAMRQ AAPL ABI ABS ABT ABX ACKH ACV ADBE ADCT ADI ADM ADP ADSK AEE AEP AES AET AFL 
AGN AIG ALD ALL ALTR AMAT AMD AMGN AMP AMR AMT AMZN AN ANDW ANF ANR AOL APA APC APD APOL 
ASH AT ATI ATML AVP AVY AXP AYE AZO BA BAC BAX BBBY BBT BBY BC BCS BDK BDX BEN BF.B BHI 
BJS BK BKS BLL BMC BMS BMY BNI BRCM BSX BT BUD BXP C CAG CAH CAM CAT CB CBH CCE CCI CCL 
CEG CELG CF CGP CHIR CHK CI CIN CINF CL CLX CMCSA CME CMI CMS CNP COF COL COP COST CPB CPWR 
CR CSC CSCO CSX CTAS CTL CTSH CTXS CVG CVX CY DD DE DELL DGX DHI DHR DIS DLTR DOV DOW DPH 
DTE DUK DVN DYN EA EBAY ECL ED EDS EFX EIX EK EL EMC EMR EOG EPG ERTS ESRX ETN ETR EXC EXPD 
F FDC FDX FHN FII FIS FISV FITB FLR FMC FNM FO FRX FSLR FTR GD GE GILD GIS GLW GM GMT GNW 
GOOG GPC GPS GR GT GW HAL HAR HCA HD HES HIG HIT HNZ HOG HON HPQ HRB HRL HSY HUM IBM ICE IGT 
INTC INTU IP IPG ITT ITW JBL JCI JDSU JEC JNJ JNS JNY JPM JWN K KBH KEY KIM KLAC KMB KMI KO 
KR KSS L LEG LEN LH LLY LM LOW LSI LUV LXK M MA MAR MAS MAT MBI MCD MCHP MCK MCO MDP MDT MET 
MGM MHP MKC MMM MO MOLX MON MOT MRK MRO MS MSFT MSI MTB MTG MWV MXIM MYL NBL NBR NDAQ NE NEE 
NEM NFLX NFX NI NKE NOC NOV NRG NSC NSM NTRS NUE NVDA NVLS NWL NWSA NYT ODP OMC ORCL ORLY OXY 
PAYX PBG PBI PCAR PCG PCL PCLN PDCO PEG PEP PFE PG PGR PH PHM PKI PLD PLL PM PNC PNW PPG PPL 
PRU PSA PTV PX QCOM QTRN R RAI RCL RHI RHT RIG RJF ROH ROK ROP RTN S SBUX SCHW SEE SGP SHW SIAL 
SJM SLB SLE SNA SNDK SNS SO SPG SPLS SRE STI STJ STR SUN SVU SWK SWN SYK SYY T TAP TDG TEK TEVA 
TGT TIF TJX TK TKR TMO TROW TRV TSN TXN TXT TYC UIS UNH UNM UNP UPS USB UTX VFC VIAB VLO VMC 
VNO VRSN VZ WAG WAT WFC WHR WLP WMB WMT WPO WRK WU WWY WY WYN X XEL XOM XRX YHOO ZBH
)

echo "================================"
echo "MULTI-TICKER PIPELINE RUNNER"
echo "================================"
echo "Tickers: ${#TICKERS[@]} stocks"
echo "Data period: $DATA_START_DATE to $END_DATE (includes warmup)"
echo "Backtest period: $BACKTEST_START_DATE to $END_DATE"
echo "Forward Horizons: ${FORWARD_HORIZONS[@]} days"
echo ""

# ============================================================================
# STAGE 1: Generate Signal Data (with all forward horizons)
# ============================================================================
if [ "$RUN_STAGE1" = true ]; then
    echo "STAGE 1: Generating signal data for all tickers..."
    echo "============================================================================"

    for ticker in "${TICKERS[@]}"; do
        echo "  Processing: $ticker"
        python3 signal_data_dump.py "$ticker" "$DATA_START_DATE" "$END_DATE" || {
            echo "    Warning: Failed to generate signals for $ticker"
        }
    done

    echo ""
    echo "✓ Stage 1 complete: Signal data generated for all tickers"
    echo ""
else
    echo "STAGE 1: Skipped (RUN_STAGE1=false)"
    echo ""
fi

if [ "$RUN_STAGE2" = true ]; then
    echo "STAGE 2: Calculating composite scores for all tickers..."
    echo "============================================================================"

    for ticker in "${TICKERS[@]}"; do
        signal_file="backtest_results/signal_data_dump/signal_data_dump_${ticker}_${DATA_START_DATE}_${END_DATE}.csv"
        
        if [ -f "$signal_file" ]; then
            echo "  Processing: $ticker"
            python3 composite_score_calculator.py "$signal_file" || {
                echo "    Warning: Failed to calculate scores for $ticker"
            }
        else
            echo "  Skipping: $ticker (signal data not found)"
        fi
    done

    echo ""
    echo "✓ Stage 2 complete: Composite scores calculated for all tickers"
    echo ""
else
    echo "STAGE 2: Skipped (RUN_STAGE2=false)"
    echo ""
fi

# ============================================================================
# STAGE 3: Run Factor Backtest for Each Forward Horizon
# ============================================================================
echo "STAGE 3: Running factor backtest for each forward horizon..."
echo "============================================================================"
echo ""

for horizon in "${FORWARD_HORIZONS[@]}"; do
    echo "Running backtest for ${horizon}-day forward horizon..."
    python3 factor_backtest_engine.py "$horizon" "$BACKTEST_START_DATE" "$END_DATE" ${TICKERS[@]} || {
        echo "  Warning: Backtest failed for ${horizon}-day horizon"
    }
    echo ""
done

echo ""
echo "================================"
echo "✓ PIPELINE COMPLETE"
echo "================================"
echo ""
echo "Results saved to:"
echo "  - backtest_results/composite_scores/      (composite score CSVs)"
echo "  - backtest_results/factor_backtest/       (backtest results)"
echo "    └── direction_score/                   (8-slide deck + metrics)"
echo "    └── magnitude_score/                   (8-slide deck + metrics)"
echo "    └── slide_07_comparison.png            (cross-factor comparison)"
echo ""