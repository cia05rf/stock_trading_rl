"""
Analysis module for evaluating trading performance.

This module provides tools for analyzing backtest results and
visualizing trading performance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.logging_config import get_logger

logger = get_logger(__name__)


def analyze_results(ledger_df: pd.DataFrame) -> Dict:
    """
    Analyze trading results from a ledger DataFrame.
    
    Args:
        ledger_df: DataFrame with trading ledger
    
    Returns:
        Dictionary with comprehensive analysis metrics including:
        - Basic metrics (trades, win rate)
        - Return metrics (total return, return %)
        - Risk-adjusted metrics (Sharpe, Sortino, Calmar)
        - Trade quality metrics (profit factor, expectancy, avg win/loss)
        - Drawdown metrics
    """
    if len(ledger_df) == 0:
        return {"error": "No trades in ledger"}

    # Normalize date column early to avoid mixed types (e.g., Timestamp + str)
    # which can break reductions like min()/max() and groupby behavior.
    ledger_df = ledger_df.copy()
    if "date" in ledger_df.columns:
        ledger_df["date"] = pd.to_datetime(ledger_df["date"], errors="coerce")
    
    # Basic metrics
    total_trades = len(ledger_df)
    buy_trades = len(ledger_df[ledger_df["action"] == "buy"])
    sell_trades = len(ledger_df[ledger_df["action"] == "sell"])
    
    # Returns
    initial_balance = ledger_df["balance_pre_trade"].iloc[0]
    final_balance = ledger_df["balance_post_trade"].iloc[-1]
    total_return = final_balance - initial_balance
    return_pct = (total_return / initial_balance) * 100
    
    # Per-trade P&L
    pnl = ledger_df["balance_post_trade"] - ledger_df["balance_pre_trade"]
    
    # Win rate
    profitable_trades = len(pnl[pnl > 0])
    losing_trades = len(pnl[pnl < 0])
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    # Profit Factor (gross profits / gross losses)
    gross_profit = pnl[pnl > 0].sum() if len(pnl[pnl > 0]) > 0 else 0
    gross_loss = abs(pnl[pnl < 0].sum()) if len(pnl[pnl < 0]) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Average Win/Loss
    avg_win = pnl[pnl > 0].mean() if len(pnl[pnl > 0]) > 0 else 0
    avg_loss = abs(pnl[pnl < 0].mean()) if len(pnl[pnl < 0]) > 0 else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # Expectancy (average profit per trade)
    expectancy = pnl.mean()
    
    # Largest win/loss
    largest_win = pnl.max() if len(pnl) > 0 else 0
    largest_loss = pnl.min() if len(pnl) > 0 else 0
    
    # Calculate daily returns for risk metrics
    daily_returns = pd.Series(dtype=float)
    sharpe_ratio = 0
    sortino_ratio = 0
    
    if "date" in ledger_df.columns:
        daily_balances = ledger_df.groupby("date")["balance_post_trade"].last()
        daily_returns = daily_balances.pct_change().dropna()
        
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            # Sharpe Ratio (annualized)
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            
            # Sortino Ratio (only penalizes downside volatility)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else daily_returns.std()
            if downside_std > 0:
                sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_std
            else:
                sortino_ratio = float('inf') if daily_returns.mean() > 0 else 0
    
    # Max drawdown
    balances = ledger_df["balance_post_trade"].values
    peak = balances[0]
    max_drawdown = 0
    max_drawdown_duration = 0
    current_drawdown_start = 0
    
    for i, balance in enumerate(balances):
        if balance > peak:
            peak = balance
            current_drawdown_start = i
        drawdown = (peak - balance) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_duration = i - current_drawdown_start
    
    # Calmar Ratio (return / max drawdown)
    calmar_ratio = return_pct / (max_drawdown * 100) if max_drawdown > 0 else float('inf')
    
    # Trade frequency
    if "date" in ledger_df.columns:
        unique_dates = ledger_df["date"].nunique()
        trades_per_day = total_trades / unique_dates if unique_dates > 0 else 0
        
        # Calculate trading days
        date_series = ledger_df["date"].dropna()
        if len(date_series) > 0:
            first_date = date_series.min()
            last_date = date_series.max()
            calendar_days = int((last_date - first_date).days)
        else:
            calendar_days = 0
    else:
        trades_per_day = 0
        unique_dates = 0
        calendar_days = 0
    
    # Action distribution
    action_counts = ledger_df["action_type"].value_counts().to_dict()
    
    # Consecutive wins/losses
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for p in pnl:
        if p > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        elif p < 0:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    return {
        # Basic metrics
        "total_trades": total_trades,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "winning_trades": profitable_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        
        # Return metrics
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return": total_return,
        "return_pct": return_pct,
        
        # Risk-adjusted metrics
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        
        # Trade quality metrics
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        
        # Drawdown metrics
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "max_drawdown_duration": max_drawdown_duration,
        
        # Streak metrics
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        
        # Activity metrics
        "trades_per_day": trades_per_day,
        "trading_days": unique_dates,
        "calendar_days": calendar_days,
        "action_distribution": action_counts,
    }


def monte_carlo_significance_test(
    ledger_df: pd.DataFrame,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
) -> Dict:
    """
    Test if strategy performance is statistically significant using Monte Carlo simulation.
    
    This test shuffles trade outcomes to simulate random timing and compares
    against actual performance to determine if results are due to skill vs luck.
    
    Args:
        ledger_df: DataFrame with trading ledger
        n_simulations: Number of random simulations to run
        confidence_level: Confidence level for significance (default 0.95)
    
    Returns:
        Dictionary with significance test results
    """
    if len(ledger_df) == 0:
        return {"error": "No trades in ledger"}
    
    pnl = (ledger_df["balance_post_trade"] - ledger_df["balance_pre_trade"]).values
    actual_total = pnl.sum()
    actual_sharpe = 0
    
    if len(pnl) > 1:
        actual_sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
    
    # Run Monte Carlo simulations
    np.random.seed(42)  # For reproducibility
    random_totals = []
    random_sharpes = []
    
    for _ in range(n_simulations):
        # Shuffle the P&L values (random entry/exit timing)
        shuffled_pnl = np.random.permutation(pnl)
        
        # Take random subset (simulating different trade selections)
        subset_size = max(1, len(shuffled_pnl) // 2)
        subset = shuffled_pnl[:subset_size]
        
        random_totals.append(subset.sum())
        if len(subset) > 1 and subset.std() > 0:
            random_sharpes.append(subset.mean() / subset.std())
    
    random_totals = np.array(random_totals)
    random_sharpes = np.array([s for s in random_sharpes if np.isfinite(s)])
    
    # Calculate p-values (proportion of random strategies that beat actual)
    p_value_return = np.mean(random_totals >= actual_total)
    p_value_sharpe = np.mean(random_sharpes >= actual_sharpe) if len(random_sharpes) > 0 else 1.0
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    return_ci_lower = np.percentile(random_totals, alpha / 2 * 100)
    return_ci_upper = np.percentile(random_totals, (1 - alpha / 2) * 100)
    
    # Determine significance
    is_significant = p_value_return < (1 - confidence_level)
    
    return {
        "actual_return": actual_total,
        "random_mean_return": np.mean(random_totals),
        "random_std_return": np.std(random_totals),
        "p_value_return": p_value_return,
        "return_ci_lower": return_ci_lower,
        "return_ci_upper": return_ci_upper,
        "actual_sharpe": actual_sharpe,
        "random_mean_sharpe": np.mean(random_sharpes) if len(random_sharpes) > 0 else 0,
        "p_value_sharpe": p_value_sharpe,
        "is_significant": is_significant,
        "confidence_level": confidence_level,
        "n_simulations": n_simulations,
        "interpretation": (
            f"Strategy returns are {'statistically significant' if is_significant else 'NOT statistically significant'} "
            f"at {confidence_level:.0%} confidence level (p={p_value_return:.4f})"
        ),
    }


def analyze_equity_curve(equity_df: pd.DataFrame) -> Dict:
    """
    Analyze equity curve including unrealized P&L.
    
    Args:
        equity_df: DataFrame with equity curve data
    
    Returns:
        Dictionary with equity analysis metrics
    """
    if len(equity_df) == 0:
        return {"error": "No equity data"}
    
    equities = equity_df["total_equity"].values
    
    # Peak and trough analysis
    peak = np.maximum.accumulate(equities)
    drawdowns = (peak - equities) / peak
    
    # Time underwater (days in drawdown)
    in_drawdown = drawdowns > 0
    underwater_periods = []
    current_underwater = 0
    
    for is_underwater in in_drawdown:
        if is_underwater:
            current_underwater += 1
        else:
            if current_underwater > 0:
                underwater_periods.append(current_underwater)
            current_underwater = 0
    
    if current_underwater > 0:
        underwater_periods.append(current_underwater)
    
    # Calculate volatility
    if "date" in equity_df.columns:
        daily_equity = equity_df.groupby("date")["total_equity"].last()
        daily_returns = daily_equity.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
    else:
        volatility = 0
    
    return {
        "start_equity": equities[0],
        "end_equity": equities[-1],
        "peak_equity": equities.max(),
        "trough_equity": equities.min(),
        "max_drawdown_pct": drawdowns.max() * 100,
        "avg_drawdown_pct": drawdowns[drawdowns > 0].mean() * 100 if any(drawdowns > 0) else 0,
        "total_days_underwater": sum(underwater_periods),
        "longest_underwater_period": max(underwater_periods) if underwater_periods else 0,
        "annualized_volatility": volatility,
        "avg_positions": equity_df["num_positions"].mean() if "num_positions" in equity_df.columns else 0,
    }


def plot_performance(
    ledger_df: pd.DataFrame,
    title: str = "Trading Performance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Create performance visualization.
    
    Args:
        ledger_df: DataFrame with trading ledger
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. Portfolio value over time
    ax = axes[0, 0]
    if "date" in ledger_df.columns:
        ledger_df["date"] = pd.to_datetime(ledger_df["date"])
        daily_balance = ledger_df.groupby("date")["balance_post_trade"].last()
        ax.plot(daily_balance.index, daily_balance.values, 'b-', linewidth=1.5)
        ax.fill_between(daily_balance.index, daily_balance.values, alpha=0.3)
    else:
        ax.plot(ledger_df["balance_post_trade"].values)
    ax.set_title("Portfolio Value")
    ax.set_ylabel("Balance (£)")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=ledger_df["balance_pre_trade"].iloc[0], color='r', linestyle='--', alpha=0.5)
    
    # 2. Trade P&L distribution
    ax = axes[0, 1]
    pnl = ledger_df["balance_post_trade"] - ledger_df["balance_pre_trade"]
    colors = ['green' if p > 0 else 'red' for p in pnl]
    ax.bar(range(len(pnl)), pnl, color=colors, alpha=0.6)
    ax.set_title("Trade P&L")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("P&L (£)")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative returns
    ax = axes[1, 0]
    cum_returns = (ledger_df["balance_post_trade"] / ledger_df["balance_pre_trade"].iloc[0] - 1) * 100
    ax.plot(cum_returns.values, 'g-', linewidth=1.5)
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Return (%)")
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # 4. Drawdown
    ax = axes[1, 1]
    balances = ledger_df["balance_post_trade"].values
    peak = np.maximum.accumulate(balances)
    drawdown = (peak - balances) / peak * 100
    ax.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.4)
    ax.plot(drawdown, 'r-', linewidth=1)
    ax.set_title("Drawdown")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    
    # 5. Action distribution
    ax = axes[2, 0]
    action_counts = ledger_df["action_type"].value_counts()
    colors_map = {'buy': 'green', 'sell': 'red', 'hold': 'gray'}
    colors = [colors_map.get(a, 'blue') for a in action_counts.index]
    ax.pie(action_counts, labels=action_counts.index, colors=colors, autopct='%1.1f%%')
    ax.set_title("Action Distribution")
    
    # 6. Monthly returns heatmap (if date available)
    ax = axes[2, 1]
    if "date" in ledger_df.columns:
        ledger_df["month"] = pd.to_datetime(ledger_df["date"]).dt.to_period('M')
        monthly_returns = ledger_df.groupby("month")["balance_post_trade"].last().pct_change() * 100
        monthly_returns = monthly_returns.dropna()
        
        if len(monthly_returns) > 0:
            colors = ['green' if r > 0 else 'red' for r in monthly_returns.values]
            ax.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
            ax.set_title("Monthly Returns")
            ax.set_xlabel("Month")
            ax.set_ylabel("Return (%)")
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
    else:
        ax.text(0.5, 0.5, "No date data", ha='center', va='center')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def compare_strategies(
    results: List[Tuple[str, pd.DataFrame]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare multiple strategy results.
    
    Args:
        results: List of (strategy_name, ledger_df) tuples
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Strategy Comparison", fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10.colors
    
    # Portfolio values
    ax = axes[0, 0]
    for i, (name, df) in enumerate(results):
        ax.plot(df["balance_post_trade"].values, label=name, color=colors[i % len(colors)])
    ax.set_title("Portfolio Value Comparison")
    ax.set_ylabel("Balance (£)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final returns bar chart
    ax = axes[0, 1]
    names = [r[0] for r in results]
    returns = [
        (r[1]["balance_post_trade"].iloc[-1] / r[1]["balance_pre_trade"].iloc[0] - 1) * 100
        for r in results
    ]
    colors_bar = ['green' if r > 0 else 'red' for r in returns]
    ax.bar(names, returns, color=colors_bar, alpha=0.7)
    ax.set_title("Total Returns")
    ax.set_ylabel("Return (%)")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Win rates
    ax = axes[1, 0]
    win_rates = []
    for name, df in results:
        pnl = df["balance_post_trade"] - df["balance_pre_trade"]
        win_rate = (pnl > 0).sum() / len(pnl) * 100 if len(pnl) > 0 else 0
        win_rates.append(win_rate)
    ax.bar(names, win_rates, color='steelblue', alpha=0.7)
    ax.set_title("Win Rate")
    ax.set_ylabel("Win Rate (%)")
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Trade counts
    ax = axes[1, 1]
    trade_counts = [len(r[1]) for r in results]
    ax.bar(names, trade_counts, color='darkorange', alpha=0.7)
    ax.set_title("Total Trades")
    ax.set_ylabel("Number of Trades")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def print_summary(metrics: dict, strategy_name: str) -> None:
    """Pretty-print core metrics to stdout."""
    def fmt_ratio(value: float) -> str:
        if value == float("inf"):
            return "∞"
        if value == float("-inf"):
            return "-∞"
        return f"{value:.2f}"
    
    print("\n" + "=" * 70)
    print(f"{strategy_name} ANALYSIS")
    print("=" * 70)
    print(f"Total Trades:        {metrics.get('total_trades', 0)}")
    print(f"Winning Trades:      {metrics.get('winning_trades', 0)}")
    print(f"Losing Trades:       {metrics.get('losing_trades', 0)}")
    print(f"Win Rate:            {metrics.get('win_rate', 0):.1%}")
    print("-" * 70)
    print(f"Initial Balance:     £{metrics.get('initial_balance', 0):,.2f}")
    print(f"Final Balance:       £{metrics.get('final_balance', 0):,.2f}")
    print(f"Total Return:        £{metrics.get('total_return', 0):,.2f} ({metrics.get('return_pct', 0):.2f}%)")
    print("-" * 70)
    print(f"Sharpe Ratio:        {fmt_ratio(metrics.get('sharpe_ratio', 0))}")
    print(f"Sortino Ratio:       {fmt_ratio(metrics.get('sortino_ratio', 0))}")
    print(f"Calmar Ratio:        {fmt_ratio(metrics.get('calmar_ratio', 0))}")
    print(f"Profit Factor:       {fmt_ratio(metrics.get('profit_factor', 0))}")
    print("-" * 70)
    print(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Expectancy:          £{metrics.get('expectancy', 0):,.2f} per trade")
    print("=" * 70 + "\n")


def generate_report(
    ledger_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    strategy_name: str = "Trading Strategy",
    include_significance_test: bool = True,
    equity_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate a comprehensive performance report.
    
    Args:
        ledger_df: DataFrame with trading ledger
        output_dir: Directory to save report files
        strategy_name: Name of the strategy
        include_significance_test: Whether to run Monte Carlo significance test
        equity_df: Optional equity curve DataFrame
    
    Returns:
        Report text
    """
    analysis = analyze_results(ledger_df)
    
    # Handle infinity values for display
    def format_ratio(value, decimals=2):
        if value == float('inf'):
            return "∞"
        elif value == float('-inf'):
            return "-∞"
        else:
            return f"{value:.{decimals}f}"
    
    report = f"""# {strategy_name} Performance Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Trades | {analysis['total_trades']:,} |
| Winning Trades | {analysis['winning_trades']:,} |
| Losing Trades | {analysis['losing_trades']:,} |
| Win Rate | {analysis['win_rate']:.1%} |
| Trading Days | {analysis.get('trading_days', 'N/A')} |

## Returns

| Metric | Value |
|--------|-------|
| Initial Balance | £{analysis['initial_balance']:,.2f} |
| Final Balance | £{analysis['final_balance']:,.2f} |
| Total Return | £{analysis['total_return']:,.2f} |
| Return % | {analysis['return_pct']:.2f}% |

## Risk-Adjusted Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sharpe Ratio | {format_ratio(analysis['sharpe_ratio'])} | {"Excellent" if analysis['sharpe_ratio'] > 2 else "Good" if analysis['sharpe_ratio'] > 1 else "Fair" if analysis['sharpe_ratio'] > 0 else "Poor"} |
| Sortino Ratio | {format_ratio(analysis['sortino_ratio'])} | Higher is better (only penalizes downside) |
| Calmar Ratio | {format_ratio(analysis['calmar_ratio'])} | Return / Max Drawdown |

## Trade Quality Metrics

| Metric | Value |
|--------|-------|
| Profit Factor | {format_ratio(analysis['profit_factor'])} |
| Expectancy | £{analysis['expectancy']:,.2f} per trade |
| Avg Win | £{analysis['avg_win']:,.2f} |
| Avg Loss | £{analysis['avg_loss']:,.2f} |
| Win/Loss Ratio | {format_ratio(analysis['win_loss_ratio'])} |
| Largest Win | £{analysis['largest_win']:,.2f} |
| Largest Loss | £{analysis['largest_loss']:,.2f} |

## Risk Metrics

| Metric | Value |
|--------|-------|
| Max Drawdown | {analysis['max_drawdown_pct']:.2f}% |
| Max DD Duration | {analysis['max_drawdown_duration']} trades |
| Max Consecutive Wins | {analysis['max_consecutive_wins']} |
| Max Consecutive Losses | {analysis['max_consecutive_losses']} |

## Trading Activity

| Action Type | Count |
|-------------|-------|
"""
    
    for action, count in analysis['action_distribution'].items():
        report += f"| {action} | {count:,} |\n"
    
    report += f"""
## Activity Summary

- Trades per day: {analysis['trades_per_day']:.2f}
- Calendar days: {analysis.get('calendar_days', 'N/A')}
"""
    
    # Add significance test if requested
    if include_significance_test:
        significance = monte_carlo_significance_test(ledger_df)
        if "error" in significance:
            report += f"""
## Statistical Significance Test

Skipped: {significance['error']}
"""
        else:
            report += f"""
## Statistical Significance Test

Monte Carlo simulation with {significance['n_simulations']:,} iterations:

| Metric | Value |
|--------|-------|
| Actual Return | £{significance['actual_return']:,.2f} |
| Random Mean | £{significance['random_mean_return']:,.2f} |
| Random Std | £{significance['random_std_return']:,.2f} |
| P-Value | {significance['p_value_return']:.4f} |
| 95% CI | [£{significance['return_ci_lower']:,.2f}, £{significance['return_ci_upper']:,.2f}] |

**{significance['interpretation']}**
"""
    
    # Add equity analysis if provided
    if equity_df is not None and len(equity_df) > 0:
        equity_analysis = analyze_equity_curve(equity_df)
        report += f"""
## Equity Curve Analysis

| Metric | Value |
|--------|-------|
| Peak Equity | £{equity_analysis['peak_equity']:,.2f} |
| Trough Equity | £{equity_analysis['trough_equity']:,.2f} |
| Avg Drawdown | {equity_analysis['avg_drawdown_pct']:.2f}% |
| Longest Underwater | {equity_analysis['longest_underwater_period']} days |
| Annualized Volatility | {equity_analysis['annualized_volatility']*100:.2f}% |
| Avg Positions | {equity_analysis['avg_positions']:.1f} |
"""
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_path = output_dir / f"{strategy_name.replace(' ', '_')}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save figure
        fig = plot_performance(ledger_df, title=strategy_name)
        fig.savefig(output_dir / f"{strategy_name.replace(' ', '_')}_performance.png", dpi=150)
        plt.close(fig)
        
        logger.info(f"Report saved to {output_dir}")
    
    return report
