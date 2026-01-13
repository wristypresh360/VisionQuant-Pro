import akshare as ak
import pandas as pd
import numpy as np
import time


class FundamentalMiner:
    def __init__(self, spot_cache_ttl_sec: int = 300, spot_retry: int = 2):
        # 缓存全市场 spot（ak.stock_zh_a_spot_em 很重，且易波动；缓存能显著降低 N/A）
        self._spot_cache_df = None
        self._spot_cache_ts = 0.0
        self._spot_cache_ttl_sec = spot_cache_ttl_sec
        self._spot_retry = spot_retry

    def get_stock_fundamentals(self, symbol):
        """
        获取深度财务指标 (含成长性与安全性分析)
        """
        symbol = str(symbol).strip().zfill(6)
        print(f"🔍 [财务分析] 正在透视 {symbol}...")

        # 默认结果结构扩展
        result = {
            "symbol": symbol,
            # 默认不要用 symbol 当 name，否则 UI 会出现 “300286(300286)” 这种重复且掩盖抓取失败
            "name": "",
            "pe_ttm": 0.0, "pb": 0.0, "total_mv": 0.0,
            "roe": 0.0, "net_profit_margin": 0.0, "asset_turnover": 0.6, "leverage": 1.0,
            "debt_asset_ratio": 0.0,
            # === 新增指标 ===
            "gross_margin": 0.0,  # 毛利率
            "current_ratio": 0.0,  # 流动比率 (偿债能力)
            "rev_growth": 0.0,  # 营收增长率
            "profit_growth": 0.0,  # 净利增长率
            "report_date": "最新"
            ,
            # === 状态字段：用于UI层判断“是否成功抓取”，避免把0当真 ===
            "_ok": {"spot": False, "finance": False},
            "_err": []
        }

        try:
            # 1. 实时估值
            try:
                spot_df = self._get_spot_df_cached(result)
                if spot_df is not None and not spot_df.empty:
                    code_col = next((c for c in spot_df.columns if '代码' in c), None)
                    if code_col:
                        target = spot_df[spot_df[code_col] == symbol]
                        if not target.empty:
                            pe_col = next((c for c in target.columns if '市盈率' in c and '动' in c), None)
                            pb_col = next((c for c in target.columns if '市净率' in c), None)
                            mv_col = next((c for c in target.columns if '总市值' in c), None)
                            name_col = next((c for c in target.columns if '名称' in c), None)

                            if pe_col:
                                result["pe_ttm"] = self._to_f(target[pe_col].values[0])
                            if pb_col:
                                result["pb"] = self._to_f(target[pb_col].values[0])
                            if mv_col:
                                result["total_mv"] = round(self._to_f(target[mv_col].values[0]) / 100000000, 2)
                            if name_col:
                                result["name"] = str(target[name_col].values[0]).strip()
                            result["_ok"]["spot"] = True
            except Exception as e:
                result["_err"].append(f"spot_df_error: {type(e).__name__}: {e}")

            # 若 spot 未拿到 name，尝试更轻量的个股信息接口兜底
            if not result.get("name"):
                try:
                    info_df = ak.stock_individual_info_em(symbol=symbol)
                    if info_df is not None and not info_df.empty:
                        # 常见字段：item/value
                        if "item" in info_df.columns and "value" in info_df.columns:
                            name_row = info_df[info_df["item"].astype(str).str.contains("股票简称|名称")]
                            if not name_row.empty:
                                result["name"] = str(name_row["value"].values[0]).strip()
                except Exception as e:
                    result["_err"].append(f"stock_individual_info_error: {type(e).__name__}: {e}")

            # 2. 深度指标：优先使用 THS 财务摘要（经验证可用；EM接口在你环境里全量报错）
            try:
                ths_df = ak.stock_financial_abstract_ths(symbol=symbol)
                if ths_df is not None and not ths_df.empty:
                    # 取最新报告期
                    if "报告期" in ths_df.columns:
                        tmp = ths_df.copy()
                        tmp["报告期_dt"] = pd.to_datetime(tmp["报告期"], errors="coerce")
                        tmp = tmp.sort_values("报告期_dt")
                        latest = tmp.iloc[-1]
                        result["report_date"] = str(latest.get("报告期", result["report_date"]))
                    else:
                        latest = ths_df.iloc[-1]

                    # 关键指标（字段名稳定）
                    result["roe"] = self._to_f(latest.get("净资产收益率"))
                    result["net_profit_margin"] = self._to_f(latest.get("销售净利率"))
                    result["gross_margin"] = self._to_f(latest.get("销售毛利率"))
                    result["current_ratio"] = self._to_f(latest.get("流动比率"))
                    result["debt_asset_ratio"] = self._to_f(latest.get("资产负债率"))
                    # 这些字段有时为 False/空，_to_f 会安全兜底为0.0
                    result["rev_growth"] = self._to_f(latest.get("营业总收入同比增长率"))
                    result["profit_growth"] = self._to_f(latest.get("净利润同比增长率"))

                    if 0 < result["debt_asset_ratio"] < 100:
                        result["leverage"] = round(1 / (1 - result["debt_asset_ratio"] / 100), 2)

                    result["_ok"]["finance"] = True
                else:
                    result["_err"].append("ths_finance_empty")
            except Exception as e:
                result["_err"].append(f"ths_finance_error: {type(e).__name__}: {e}")

            # 3. 若仍拿不到 ROE，则用 PB/PE 推算（标注为推算，不再默默写0）
            if not result["_ok"]["finance"] and result["pe_ttm"] > 0:
                result["roe"] = round((result["pb"] / result["pe_ttm"]) * 100, 2)
                # 只作为兜底推算，不写入 _ok.finance
                result["_err"].append("roe_estimated_by_pb_pe")

        except Exception as e:
            result["_err"].append(f"spot_error: {type(e).__name__}: {e}")
            print(f"⚠️ 财报异常: {e}")

        return result

    def _get_spot_df_cached(self, result: dict):
        """
        获取全市场 spot 数据（带缓存 + 重试）。
        """
        now = time.time()
        if self._spot_cache_df is not None and (now - self._spot_cache_ts) < self._spot_cache_ttl_sec:
            return self._spot_cache_df

        last_err = None
        for i in range(max(1, self._spot_retry + 1)):
            try:
                df = ak.stock_zh_a_spot_em()
                if df is None or df.empty:
                    raise RuntimeError("spot_df_empty")
                # 标准化代码列为6位
                code_col = next((c for c in df.columns if '代码' in c), None)
                if code_col:
                    df[code_col] = df[code_col].astype(str).str.zfill(6)
                self._spot_cache_df = df
                self._spot_cache_ts = now
                return df
            except Exception as e:
                last_err = e
                # 轻微退避，降低瞬时波动/限流影响
                time.sleep(0.25 * (i + 1))

        if last_err is not None:
            result["_err"].append(f"spot_retry_failed: {type(last_err).__name__}: {last_err}")
        return None

    # ... (get_industry_peers, _find_val, _to_f 保持不变，直接复用原有的即可) ...
    # 为了完整性，这里简写保留辅助函数结构
    def get_industry_peers(self, symbol):
        # (复用之前的代码逻辑)
        symbol = str(symbol).strip().zfill(6)
        try:
            info_df = ak.stock_individual_info_em(symbol=symbol)
            industry = info_df[info_df['item'] == '行业']['value'].values[0]
            full_market = ak.stock_zh_a_spot_em()

            if '行业' in full_market.columns:
                peers_df = full_market[full_market['行业'] == industry].copy()
            else:
                industry_cons = ak.stock_board_industry_cons_em(symbol=industry)
                peer_codes = industry_cons['代码'].astype(str).str.zfill(6).tolist()
                full_market['代码'] = full_market['代码'].astype(str).str.zfill(6)
                peers_df = full_market[full_market['代码'].isin(peer_codes)].copy()

            mkt_cap_col = [c for c in peers_df.columns if '市值' in c][0]
            peers_df = peers_df.sort_values(by=mkt_cap_col, ascending=False).head(6).copy()

            comparison_df = pd.DataFrame({
                "代码": peers_df['代码'].astype(str).str.zfill(6),
                "名称": peers_df['名称'],
                "PE(动)": peers_df['市盈率-动态'].apply(self._to_f),
                "PB": peers_df['市净率'].apply(self._to_f),
                "市值(亿)": (peers_df[mkt_cap_col].apply(self._to_f) / 100000000).round(2)
            })
            comparison_df['ROE(推算%)'] = np.where(comparison_df['PE(动)'] > 0,
                                                   (comparison_df['PB'] / comparison_df['PE(动)'] * 100).round(2), 0)
            return industry, comparison_df
        except:
            return "未知", pd.DataFrame()

    def _find_val(self, row, cols, keywords):
        for c in cols:
            if all(k in str(c) for k in keywords): return self._to_f(row[c])
        return 0.0

    def _to_f(self, val):
        try:
            if val is None or str(val) in ['-', 'nan', '']: return 0.0
            return float(str(val).replace('%', '').replace(',', ''))
        except:
            return 0.0