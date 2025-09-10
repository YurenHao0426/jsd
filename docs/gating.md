# Gating Rules (w_t) for Bias-aware Training/Eval

## Purpose
仅在**与性别相关**的时间步上启用损失或统计，避免“误中和”（例如语法必然一致的指代）。

## Tokenization Considerations
- 词表来自 `assets/groups/en_*.txt` 与 `assets/triggers/occupations_en.txt`。
- 将词表映射到 tokenizer 的 **token-id 集**；注意 Qwen/BPE 常区分“空格前缀”子词（如 `" he"` vs `"he"`）。
- 若一个词被拆为多 token，**组质量统计**建议以**首 token**为代表（工程近似，避免乘积概率偏置）。

## Triggers (either condition fires)
1. **Top-K 触发**：若某步的 top-\(K\)（建议 \(K=20\)）候选 token 命中 \(F \cup M\)，则 \(w_t=1\)。
2. **职业+代词窗口**：若输入窗口内（若干 token）同时出现
   - 一个 `occupations_en.txt` 中的职业词；
   - **以及** 代词/姓名（代词来自 \(F\cup M\)，姓名可用 `weat_*_names.txt` 辅助检测），
   则在该区域内相邻若干步打开 \(w_t\)。

## Window of Application
- 从触发点起的 **后 \(W\) 个步**生效，默认 \(W=3\)：
\[
w_{t'}=1,\quad \forall t'\in\{t+1,\dots,t+W\}.
\]
- 评测与训练均使用相同窗口与 \(K\)。

## Exclusions / Heuristics
- **强语法一致性**（如 “Mary said she …”）默认**只观测不训练**（避免把正确一致性中和）。
- 消歧：`her` vs `here`；标点黏连（`her,`）→ 以 tokenizer 分词为准。
- 敬称含点与不含点（`mr`/`mr.`、`ms`/`ms.`）都纳入词表。

## Logging (for sanity)
- 记录：触发类型分布（Top-K vs 职业窗口）、覆盖率 \(\sum_t w_t/\text{all steps}\)、每批次平均触发步数。
- 若覆盖率过低（<5%），适当增大 \(K\) 或 \(W\)；过高（>40%）则下调以避免过度训练。
