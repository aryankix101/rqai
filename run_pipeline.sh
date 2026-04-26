# Pipeline runner for multi-ticker backtesting

set -e 

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_CMD="${PYTHON_BIN}"
elif [ -x "./.venv/bin/python" ]; then
    PYTHON_CMD="./.venv/bin/python"
else
    PYTHON_CMD="python3"
fi

# Configuration
DATA_START_DATE="1997-01-01"      
BACKTEST_START_DATE="2000-01-01"  
END_DATE="2025-12-31"
FORWARD_HORIZONS=(5 10 20 30 60)
RUN_STAGE1=true                  
RUN_STAGE2=true                  

# Ticker list
TICKERS=(
A AABA AAL AAMRQ AAPL ABI ABS ABT ABX ACKH ACV ADBE ADCT ADI ADM ADP ADSK AEE AEP AES AET AFL AGN AIG ALD ALL ALTR AMAT AMD AMGN AMP AMR AMT AMZN AN ANDW ANF ANR AOL AON APA APC APD APOL ARNC ASH AT ATI ATML AVP AVY AXP AYE AZO BA BAC BAX BAY BBBY BBI BBT BBY BC BCO BCR BCS BDK BDX BEAM BEN BEV BF-B BFI BFO BGG BHGE BHI BHMSQ BJS BK BKB BKS BLL BLS BLY BMC BMET BMGCA BMS BMY BNI BNL BOAT BOL BR BRCM BSX BT BUD BXP C CAG CAH CAM CAT CB CBB CBE CBH CBS CCE CCI CCL CEG CELG CF CFL CG CGP CHA CHIR CHK CHRS CI CIN CINF CL CLX CMCSA CME CMI CMS CNP COF COL COP COST CPB CR CSC CSCO CSX CTAS CTL CTSH CTXS CVG CVX CY DALRQ DCNAQ DD DE DEC DELL DGN DGX DHI DHR DI DIGI DIS DLTR DLX DOV DOW DPH DRI DTE DUK DVN DWD DXC DYN EA EBAY EC ECH ECL ECO ED EDS EFU EFX EHC EIX EK EKDKQ EL EMC EMN EMR ENRNQ ENS EOG EPG ERTS ESRX ETN ETR ETS EXC EXPD F FBF FCN FCX FDC FDX FE FG FHN FII FIS FISV FITB FJ FL FLMIQ FLR FLTWQ FMC FMCC FNM FNMA FO FRO FRX FSLR FTL FTR FWLT G GAPTQ GAS GD GDW GE GFS GIDL GILD GIS GLW GM GMT GNT GNW GOOG GP GPC GPS GPU GR GRA GRN GSX GT GTE GW GWF GWW HAL HAR HBAN HCA HD HES HET HFS HI HIG HIT HLT HM HNZ HOG HON HPH HPQ HRB HRL HRS HSH HSY HUM IBM ICE IFF IGT IKN INCLF INGR INTC INTU IP IPG IR ITT ITW JAVA JBL JCI JCP JDSU JEC JH JNJ JNS JNY JOS JP JPM JWN K KATE KBH KEY KIM KLAC KM KMB KMG KMI KO KR KRB KRI KSS KWP L LB LDG LDW LEG LEHMQ LEN LH LLX LLY LM LNC LOW LPX LSI LU LUV LXK M MA MAR MAS MAT MBI MCD MCHP MCK MCO MDP MDT MEE MEL MER MET MGM MHP MII MIL MIR MKC MKG MMM MNR MO MOB MOLX MON MOT MRK MRO MS MSFT MSI MST MTB MTG MTLQQ MU MWI MWV MXIM MYG MYL MZIAQ NAE NAV NBL NBR NC NCC NDAQ NE NEE NEM NFLX NFX NI NKE NLC NLV NMK NOC NOV NOVL NRG NRTLQ NSC NSI NSM NTRS NUE NVDA NVLS NWL NWSA NYN NYT OAT ODP OKE OMC OMX ONE ORCL ORLY ORX OWENQ OXY PAC PAS PAYX PBG PBI PBY PCAR PCG PCH PCL PCLN PD PDCO PDG PEG PEL PEP PET PFE PG PGL PGN PGR PH PHA PHB PHM PKI PLD PLL PM PNC PNU PNW PPG PPL PPW PRD PRU PSA PTV PVN PX PZE QCOM QTRN R RAD RAI RAL RBD RBK RCL RDC RDS RHI RHT RIG RJF RLM RML RNB ROH ROK ROP RRD RSHCQ RTN RX RYAN RYC RYI S SAF SB SBUX SCHW SCI SEE SEG SFA SFS SGID SGP SHN SHW SIAL SJM SK SLB SLE SMI SMS SNA SNDK SNS SNT SO SPG SPGI SPLS SRE SRR STI STJ STO STR SUN SVU SWK SWN SYK SYY T TAP TDG TDM TEK TEN TEVA TGNA TGT THC THY TIF TIN TJX TK TKR TLAB TMC TMK TMO TNB TOY TRB TROW TRV TRW TSN TUP TWX TX TXN TXT TXU TYC UA UCC UCL UCM UIS UK UMG UN UNH UNM UNP UPR UPS USB USBC USH USS UST USW UTX VAT VFC VIAB VLO VMC VNO VO VRSN VZ WAG WAI WAT WB WBA WCOEQ WEN WFC WHR WLA WLL WLP WMB WMT WMX WNDXQ WOR WPO WRK WU WWY WY WYE WYN X XEL XOM XRX YHOO YRCW ZBH
)

echo "================================"
echo "MULTI-TICKER PIPELINE RUNNER"
echo "================================"
echo "Tickers: ${#TICKERS[@]} stocks"
echo "Data period: $DATA_START_DATE to $END_DATE (includes warmup)"
echo "Backtest period: $BACKTEST_START_DATE to $END_DATE"
echo "Forward Horizons: ${FORWARD_HORIZONS[@]} days"
echo "Python: $PYTHON_CMD"
echo ""

# ============================================================================
# STAGE 1: Generate Signal Data (with all forward horizons)
# ============================================================================
if [ "$RUN_STAGE1" = true ]; then
    echo "STAGE 1: Generating signal data for all tickers..."
    echo "============================================================================"

    for ticker in "${TICKERS[@]}"; do
        echo "  Processing: $ticker"
        "$PYTHON_CMD" signal_data_dump.py "$ticker" "$DATA_START_DATE" "$END_DATE" || {
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
            "$PYTHON_CMD" composite_score_calculator.py "$signal_file" || {
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
    "$PYTHON_CMD" factor_backtest_engine.py "$horizon" "$BACKTEST_START_DATE" "$END_DATE" ${TICKERS[@]} || {
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
