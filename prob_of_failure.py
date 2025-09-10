import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)  # For reproducibility

def futures_portfolio_simulation(
    initial_capital=100.0,
    leverage=1.6,
    mu_annual=0.10,           # Updated: 10% annual return
    sigma_annual=0.10,        # Updated: 10% annual volatility  
    years=5,
    trading_days=252,
    init_margin_rate=0.137,   # Updated: 13.7% initial margin
    maint_margin_rate=0.122,  # Updated: 12.2% maintenance margin
    funding_rate_annual=0.068, # 6.8% funding rate
    t_df=4,                   # t-distribution degrees of freedom
    simulations=10000,
    transaction_cost_bp=3,
    save_sample_paths=50
):
    """
    Monte Carlo simulation for futures portfolio with margin mechanics.
    - Borrowing and interest only occur during margin calls
    - Fixed notional exposure (no daily rebalancing)
    - t-distribution based returns with variance normalization
    """
    
    # Calculate derived parameters
    notional = leverage * initial_capital
    mu_daily = mu_annual / trading_days
    sigma_daily = sigma_annual / np.sqrt(trading_days)
    funding_daily = funding_rate_annual / trading_days
    transaction_daily = transaction_cost_bp / 10000 / trading_days
    
    # Margin thresholds
    initial_required = init_margin_rate * notional
    maintenance_required = maint_margin_rate * notional
    
    # Results storage
    results = {
        'failures': 0,
        'margin_call_paths': 0,
        'total_margin_calls': 0,
        'final_equities': [],
        'max_drawdowns': [],
        'total_interests': [],
        'sample_paths': []
    }
    
    print(f"Simulation Configuration:")
    print(f"- Leverage: {leverage}x")
    print(f"- Notional: {notional:.1f}")
    print(f"- Initial margin: {initial_required:.1f} ({init_margin_rate:.1%})")
    print(f"- Maintenance margin: {maintenance_required:.1f} ({maint_margin_rate:.1%})")
    print(f"- Daily expected return: {mu_daily:.4%}")
    print(f"- Daily volatility: {sigma_daily:.4%}")
    
    # Run simulations
    for sim in range(simulations):
        # Initialize each simulation path
        equity = initial_capital
        debt = 0.0
        peak_equity = initial_capital
        max_drawdown = 0.0
        margin_calls_count = 0
        total_interest = 0.0
        failed = False
        
        # Sample path storage
        if sim < save_sample_paths:
            path_data = {
                'days': [0],
                'equity': [equity],
                'debt': [debt]
            }
        
        # Daily simulation loop
        for day in range(1, trading_days * years + 1):
            # Generate t-distribution return with variance normalization
            z = stats.t.rvs(df=t_df)
            z *= np.sqrt((t_df - 2) / t_df)  # Normalize variance to 1
            daily_return = mu_daily + sigma_daily * z
            
            # Calculate daily P&L and costs
            daily_pnl = notional * daily_return
            daily_cost = notional * transaction_daily
            
            # Update equity
            equity += daily_pnl - daily_cost
            
            # Check for immediate failure
            if equity <= 0:
                results['failures'] += 1
                failed = True
                break
            
            # Margin call check
            if equity < maintenance_required:
                shortfall = initial_required - equity
                if shortfall > 0:
                    debt += shortfall
                    equity += shortfall
                    margin_calls_count += 1
                    
                    if equity <= 0:
                        results['failures'] += 1
                        failed = True
                        break
            
            # Interest accrual only when debt exists
            if debt > 0:
                daily_interest = debt * funding_daily
                debt += daily_interest  # Compound debt growth
                equity -= daily_interest
                total_interest += daily_interest
                
                if equity <= 0:
                    results['failures'] += 1
                    failed = True
                    break
            
            # Debt repayment with excess funds
            if debt > 0 and equity > initial_required * 1.1:
                repayment = min(debt, equity - initial_required)
                debt -= repayment
                equity -= repayment
            
            # Track maximum drawdown
            if equity > peak_equity:
                peak_equity = equity
            current_drawdown = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Sample path storage
            if sim < save_sample_paths and day % 10 == 0:
                path_data['days'].append(day)
                path_data['equity'].append(equity)
                path_data['debt'].append(debt)
        
        # Store results for completed paths
        if not failed:
            results['final_equities'].append(equity)
            results['max_drawdowns'].append(max_drawdown)
            results['total_interests'].append(total_interest)
            
            if margin_calls_count > 0:
                results['margin_call_paths'] += 1
                results['total_margin_calls'] += margin_calls_count
        
        # Store sample path data
        if sim < save_sample_paths:
            path_data['failed'] = failed
            path_data['margin_calls'] = margin_calls_count
            results['sample_paths'].append(path_data)
        
        # Progress reporting
        if (sim + 1) % 1000 == 0:
            current_failure_rate = results['failures'] / (sim + 1) * 100
            print(f"Progress: {sim+1:,}/{simulations:,} ({(sim+1)/simulations*100:.1f}%) - "
                  f"Current failure rate: {current_failure_rate:.3f}%")
    
    return results

def validate_historical_mdd(results, historical_mdd=0.32, leverage=1.6):
    """Validate simulation results against Historical MDD"""
    
    print(f"\n{'='*70}")
    print(f"Historical MDD {historical_mdd:.0%} Validation (Leverage {leverage}x)")
    print(f"{'='*70}")
    
    if not results['max_drawdowns']:
        print("No surviving portfolios for validation")
        return "NO_DATA"
    
    mdd_data = np.array(results['max_drawdowns']) * 100
    historical_mdd_pct = historical_mdd * 100
    
    # Calculate statistics
    stats_dict = {
        'mean': np.mean(mdd_data),
        'median': np.median(mdd_data),
        'std': np.std(mdd_data),
        'p75': np.percentile(mdd_data, 75),
        'p90': np.percentile(mdd_data, 90),
        'p95': np.percentile(mdd_data, 95),
        'p99': np.percentile(mdd_data, 99),
        'max': np.max(mdd_data)
    }
    
    print(f"Simulation MDD Distribution:")
    print(f"   • Mean: {stats_dict['mean']:.1f}% | Median: {stats_dict['median']:.1f}%")
    print(f"   • Std Dev: {stats_dict['std']:.1f}% | Max: {stats_dict['max']:.1f}%")
    print(f"   • Percentiles: 75%={stats_dict['p75']:.1f}% | 90%={stats_dict['p90']:.1f}% | 95%={stats_dict['p95']:.1f}% | 99%={stats_dict['p99']:.1f}%")
    
    # Historical MDD position analysis
    exceed_count = np.sum(mdd_data > historical_mdd_pct)
    exceed_rate = exceed_count / len(mdd_data) * 100
    percentile_position = 100 - exceed_rate
    
    print(f"\nHistorical MDD Position Analysis:")
    print(f"   • Paths exceeding Historical MDD: {exceed_count:,} ({exceed_rate:.1f}%)")
    print(f"   • Historical MDD corresponds to {percentile_position:.1f}% percentile")
    
    # Validation assessment
    print(f"\nValidation Results:")
    if 80 <= percentile_position <= 95:
        validation_result = "EXCELLENT"
        print(f"   EXCELLENT: Historical MDD at {percentile_position:.1f}% percentile")
        print(f"   → Simulation model accurately reflects actual market risk")
    elif 70 <= percentile_position < 80 or 95 < percentile_position <= 98:
        validation_result = "GOOD"
        print(f"   GOOD: Historical MDD at {percentile_position:.1f}% percentile")
        print(f"   → Simulation model is generally appropriate")
    else:
        validation_result = "WARNING"
        print(f"   WARNING: Historical MDD at {percentile_position:.1f}% percentile")
        print(f"   → Consider model parameter adjustments")
    
    return validation_result, stats_dict, percentile_position

def create_comprehensive_plots(results_16x, results_24x):
    """Create comprehensive visualization of results"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Portfolio paths
    ax1 = plt.subplot(3, 3, 1)
    for path in results_16x['sample_paths'][:20]:
        color = 'red' if path['failed'] else 'blue'
        alpha = 0.8 if path['failed'] else 0.4
        years_axis = np.array(path['days']) / 252
        plt.plot(years_axis, path['equity'], color=color, alpha=alpha, linewidth=1)
    
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.axhline(y=21.92, color='green', linestyle=':', alpha=0.7, label='Initial Margin')
    plt.axhline(y=19.52, color='red', linestyle='-.', alpha=0.7, label='Maintenance Margin')
    plt.title('Leverage 1.6x - Portfolio Paths', fontweight='bold')
    plt.xlabel('Years')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    for path in results_24x['sample_paths'][:20]:
        color = 'red' if path['failed'] else 'orange'
        alpha = 0.8 if path['failed'] else 0.4
        years_axis = np.array(path['days']) / 252
        plt.plot(years_axis, path['equity'], color=color, alpha=alpha, linewidth=1)
    
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.axhline(y=32.88, color='green', linestyle=':', alpha=0.7, label='Initial Margin')
    plt.axhline(y=29.28, color='red', linestyle='-.', alpha=0.7, label='Maintenance Margin')
    plt.title('Leverage 2.4x - Portfolio Paths', fontweight='bold')
    plt.xlabel('Years')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Failure probability comparison
    ax3 = plt.subplot(3, 3, 3)
    leverages = ['1.6x', '2.4x']
    failure_rates = [
        results_16x['failures'] / 10000 * 100,
        results_24x['failures'] / 10000 * 100
    ]
    bars = plt.bar(leverages, failure_rates, color=['skyblue', 'salmon'], alpha=0.8)
    plt.title('5-Year Failure Probability', fontweight='bold')
    plt.ylabel('Failure Probability (%)')
    
    for i, v in enumerate(failure_rates):
        plt.text(i, v + max(0.01, v*0.1), f'{v:.3f}%', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Final equity distribution
    ax4 = plt.subplot(3, 3, 4)
    if results_16x['final_equities']:
        plt.hist(results_16x['final_equities'], bins=50, alpha=0.7, 
                label=f'1.6x (n={len(results_16x["final_equities"]):,})', 
                color='skyblue', density=True)
    if results_24x['final_equities']:
        plt.hist(results_24x['final_equities'], bins=50, alpha=0.7, 
                label=f'2.4x (n={len(results_24x["final_equities"]):,})', 
                color='salmon', density=True)
    
    plt.axvline(x=100, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.title('Final Equity Distribution', fontweight='bold')
    plt.xlabel('Final Equity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. MDD distribution with Historical MDD
    ax5 = plt.subplot(3, 3, 5)
    if results_16x['max_drawdowns']:
        mdd_16x = [dd * 100 for dd in results_16x['max_drawdowns']]
        plt.hist(mdd_16x, bins=30, alpha=0.7, label='1.6x', color='skyblue', density=True)
    if results_24x['max_drawdowns']:
        mdd_24x = [dd * 100 for dd in results_24x['max_drawdowns']]
        plt.hist(mdd_24x, bins=30, alpha=0.7, label='2.4x', color='salmon', density=True)
    
    plt.axvline(x=32, color='red', linestyle='--', linewidth=2, label='Historical MDD 32%')
    plt.title('Maximum Drawdown Distribution', fontweight='bold')
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Margin call rates
    ax6 = plt.subplot(3, 3, 6)
    margin_rates = [
        results_16x['margin_call_paths'] / 10000 * 100,
        results_24x['margin_call_paths'] / 10000 * 100
    ]
    bars = plt.bar(leverages, margin_rates, color=['lightblue', 'lightcoral'], alpha=0.8)
    plt.title('Margin Call Experience Rate', fontweight='bold')
    plt.ylabel('Margin Call Rate (%)')
    
    for i, v in enumerate(margin_rates):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 6. Interest burden
    ax7 = plt.subplot(3, 3, 7)
    if results_16x['total_interests']:
        plt.hist(results_16x['total_interests'], bins=30, alpha=0.7, 
                label='1.6x', color='skyblue', density=True)
    if results_24x['total_interests']:
        plt.hist(results_24x['total_interests'], bins=30, alpha=0.7, 
                label='2.4x', color='salmon', density=True)
    
    plt.title('Cumulative Interest Distribution', fontweight='bold')
    plt.xlabel('Total Interest Paid')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Risk vs Return scatter
    ax8 = plt.subplot(3, 3, 8)
    if results_16x['final_equities'] and results_16x['max_drawdowns']:
        returns_16x = [(eq/100 - 1) * 100 for eq in results_16x['final_equities']]
        mdd_16x = [dd * 100 for dd in results_16x['max_drawdowns']]
        plt.scatter(mdd_16x, returns_16x, alpha=0.5, s=10, color='blue', label='1.6x')
    
    if results_24x['final_equities'] and results_24x['max_drawdowns']:
        returns_24x = [(eq/100 - 1) * 100 for eq in results_24x['final_equities']]
        mdd_24x = [dd * 100 for dd in results_24x['max_drawdowns']]
        plt.scatter(mdd_24x, returns_24x, alpha=0.5, s=10, color='red', label='2.4x')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=32, color='red', linestyle='--', alpha=0.5, label='Historical MDD')
    plt.title('Risk vs Return Profile', fontweight='bold')
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Total Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_data = []
    for name, results in [('1.6x', results_16x), ('2.4x', results_24x)]:
        failure_rate = results['failures'] / 10000 * 100
        margin_rate = results['margin_call_paths'] / 10000 * 100
        avg_final = np.mean(results['final_equities']) if results['final_equities'] else 0
        avg_mdd = np.mean(results['max_drawdowns']) * 100 if results['max_drawdowns'] else 0
        avg_interest = np.mean(results['total_interests']) if results['total_interests'] else 0
        
        stats_data.append([
            name,
            f"{failure_rate:.3f}%",
            f"{margin_rate:.1f}%",
            f"{avg_final:.0f}",
            f"{avg_mdd:.1f}%",
            f"{avg_interest:.2f}"
        ])
    
    table = ax9.table(
        cellText=stats_data,
        colLabels=['Leverage', 'Failure Rate', 'MC Rate', 'Avg Final', 'Avg MDD', 'Avg Interest'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    ax9.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

def print_detailed_results(name, results):
    """Print comprehensive results analysis"""
    print(f"\n{'='*60}")
    print(f"{name} Simulation Results")
    print(f"{'='*60}")
    
    failure_rate = results['failures'] / 10000 * 100
    survival_rate = 100 - failure_rate
    margin_rate = results['margin_call_paths'] / 10000 * 100
    
    print(f"Basic Statistics:")
    print(f"   • 5-year failure probability: {failure_rate:.3f}%")
    print(f"   • Survival rate: {survival_rate:.3f}%")
    print(f"   • Margin call experience rate: {margin_rate:.2f}%")
    
    if results['margin_call_paths'] > 0:
        avg_calls = results['total_margin_calls'] / results['margin_call_paths']
        print(f"   • Average margin calls when occurred: {avg_calls:.2f} times")
    
    if results['final_equities']:
        final_eq = results['final_equities']
        print(f"\nSurvivor Portfolio Performance:")
        print(f"   • Mean final equity: {np.mean(final_eq):.1f}")
        print(f"   • Median final equity: {np.median(final_eq):.1f}")
        print(f"   • Standard deviation: {np.std(final_eq):.1f}")
        print(f"   • Range: {np.min(final_eq):.1f} to {np.max(final_eq):.1f}")
        
        returns = [(eq/100 - 1) * 100 for eq in final_eq]
        print(f"\nReturn Statistics:")
        print(f"   • Mean return: {np.mean(returns):.1f}%")
        print(f"   • Median return: {np.median(returns):.1f}%")
        print(f"   • Positive return rate: {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}%")
    
    if results['max_drawdowns']:
        mdd_data = [dd * 100 for dd in results['max_drawdowns']]
        print(f"\nDrawdown Statistics:")
        print(f"   • Mean MDD: {np.mean(mdd_data):.1f}%")
        print(f"   • Median MDD: {np.median(mdd_data):.1f}%")
        print(f"   • Worst MDD: {np.max(mdd_data):.1f}%")
        print(f"   • 95th percentile MDD: {np.percentile(mdd_data, 95):.1f}%")

# Execute simulation
def run_complete_simulation():
    """Execute complete simulation with validation"""
    
    print("Monte Carlo Simulation with Updated Parameters")
    print("Annual return: 10%, Annual volatility: 10%")
    print("Initial margin: 13.7%, Maintenance margin: 12.2%")
    
    # Common parameters
    common_params = {
        'mu_annual': 0.10,
        'sigma_annual': 0.10,
        'init_margin_rate': 0.137,
        'maint_margin_rate': 0.122,
        'funding_rate_annual': 0.068,
        'years': 5,
        'simulations': 10000
    }
    
    print("\n1️⃣ Running Leverage 1.6x Simulation...")
    results_16x = futures_portfolio_simulation(leverage=1.6, **common_params)
    
    print("\n2️⃣ Running Leverage 2.4x Simulation...")
    results_24x = futures_portfolio_simulation(leverage=2.4, **common_params)
    
    # Print detailed results
    print_detailed_results("Leverage 1.6x", results_16x)
    print_detailed_results("Leverage 2.4x", results_24x)
    
    # Historical MDD validation
    validation_16x, stats_16x, percentile_16x = validate_historical_mdd(
        results_16x, historical_mdd=0.32, leverage=1.6
    )
    
    # 2.4x MDD analysis
    if results_24x['max_drawdowns']:
        actual_mean_mdd_24x = np.mean(results_24x['max_drawdowns']) * 100
        expected_mdd_24x = 32 * (2.4 / 1.6)  # Linear scaling: 48%
        
        print(f"\nLeverage 2.4x MDD Analysis:")
        print(f"   • Expected MDD (linear scaling): {expected_mdd_24x:.1f}%")
        print(f"   • Actual simulation mean MDD: {actual_mean_mdd_24x:.1f}%")
        print(f"   • Difference: {actual_mean_mdd_24x - expected_mdd_24x:+.1f}%")
    
    # Risk comparison
    print(f"\n{'='*60}")
    print("Leverage Risk Comparison")
    print(f"{'='*60}")
    
    if results_16x['failures'] > 0 and results_24x['failures'] > 0:
        risk_multiplier = results_24x['failures'] / results_16x['failures']
    else:
        risk_multiplier = "N/A (insufficient failures)"
    
    margin_risk_ratio = results_24x['margin_call_paths'] / max(results_16x['margin_call_paths'], 1)
    
    print(f"• Failure risk multiplier: {risk_multiplier}")
    print(f"• Margin call risk multiplier: {margin_risk_ratio:.1f}x")
    
    if results_16x['final_equities'] and results_24x['final_equities']:
        avg_return_16x = np.mean([(eq/100-1)*100 for eq in results_16x['final_equities']])
        avg_return_24x = np.mean([(eq/100-1)*100 for eq in results_24x['final_equities']])
        print(f"• Average returns: 1.6x = {avg_return_16x:.1f}% vs 2.4x = {avg_return_24x:.1f}%")
    
    # Final recommendations
    print(f"\nFinal Investment Recommendations:")
    
    if validation_16x == "EXCELLENT":
        print(f"✅ Simulation model excellently validated against historical data")
        print(f"   → Leverage 1.6x: Suitable for medium-term investment")
        print(f"     - Must tolerate potential 32% drawdowns")
        print(f"     - Very low failure probability with current parameters")
        print(f"   → Leverage 2.4x: Consider for shorter-term aggressive strategies")
        print(f"     - Higher margin call frequency but still manageable")
        print(f"     - Expect potential drawdowns up to ~40-50%")
    else:
        print(f"⚠️  Model validation suggests parameter review may be needed")
    
    # Generate comprehensive visualization
    print(f"\nGenerating comprehensive charts...")
    create_comprehensive_plots(results_16x, results_24x)
    
    return results_16x, results_24x

# Execute the simulation
if __name__ == "__main__":
    results_16x, results_24x = run_complete_simulation()
    print("\n✅ Simulation completed successfully!")
