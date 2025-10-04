# Phase 3: Future Improvements for DB Agent

## Current Status
- **Phase 1 (with Q66 fix)**: 36/43 expected (83.7%)
- **Remaining Issues**: 7 questions requiring advanced features

## Issues to Address

### 1. Complex Multi-Step Calculations (Q69-Q72)

#### Q69: CAGR Calculation
**Question**: "What is the CAGR of coal electricity in the United States from 2000 to 2020?"
**Expected**: 32.464
**Current Issue**: Requires compound annual growth rate formula

**Solution Approach**:
```python
# Add CAGR pattern to SQL prompt
"""
- CAGR (Compound Annual Growth Rate):
  CAGR = (((end_value / start_value) ^ (1 / years)) - 1) * 100

  SQL Implementation:
  SELECT (POWER((end_val / NULLIF(start_val, 0)), (1.0 / num_years)) - 1) * 100

  Example:
  SELECT (
    POWER(
      (SELECT coal_electricity FROM table WHERE year=2020) /
      NULLIF((SELECT coal_electricity FROM table WHERE year=2000), 0),
      1.0 / 20
    ) - 1
  ) * 100 AS cagr
"""
```

#### Q70: Multi-Country Ratio Calculation
**Question**: Ratio comparison involving multiple countries
**Current Issue**: Complex aggregation + ratio calculation

**Solution Approach**:
- Add CTE (Common Table Expression) examples
- Show how to calculate ratios with aggregated values
```sql
WITH agg AS (
  SELECT SUM(col1) as total1, SUM(col2) as total2
  FROM table WHERE country IN ('X', 'Y')
)
SELECT (total1 / NULLIF(total2, 0)) AS ratio FROM agg
```

#### Q71-Q72: Percentage Change Calculations
**Question**: Temporal percentage changes
**Current Issue**: Requires percentage change formula

**Solution Approach**:
```python
# Add percentage change pattern
"""
- Percentage Change:
  ((new_value - old_value) / old_value) * 100

  SQL:
  SELECT (
    ((SELECT val FROM table WHERE year=2020) - (SELECT val FROM table WHERE year=2015)) /
    NULLIF((SELECT val FROM table WHERE year=2015), 0)
  ) * 100 AS pct_change
"""
```

### 2. Column/Data Issues

#### Q51: Rounding Precision
**Expected**: 32.134
**Predicted**: 32.129
**Difference**: 0.005

**Solution Approach**:
- Check if database values need rounding
- Verify AVG() calculation includes correct rows
- May need to adjust tolerance or investigate data source

#### Q52: Magnitude Error
**Expected**: 56.687
**Predicted**: 6.687
**Factor**: ~8.5x difference

**Solution Approach**:
- Likely wrong year range for average calculation
- Check the exact years in the question vs SQL query
- Verify column name matches question intent

### 3. Data Availability

#### Q59: GDP Data
**Question**: GDP-related query
**Current**: Routes to hybrid/Wikipedia (not implemented)

**Solution Approach**:
- Implement Wikipedia agent for GDP lookups
- Implement hybrid mode to combine DB + Wikipedia
- Add fallback handling when data not in database

## Implementation Roadmap

### Priority 1: Multi-Step Query Handler
Add advanced SQL patterns to the prompt:
1. CAGR calculations with POWER()
2. Percentage change formulas
3. CTE usage for complex aggregations
4. Nested subqueries with ratio calculations

**Estimated Impact**: +3-4 questions (Q69-Q72)

### Priority 2: Column Validation
Add pre-execution validation:
1. Check if columns exist in schema before generating SQL
2. Validate year ranges match question
3. Add column name disambiguation (e.g., "coal CO2" vs "coal electricity")

**Estimated Impact**: +1-2 questions (Q51, Q52)

### Priority 3: Hybrid Data Sources
Implement missing agent types:
1. Complete Wikipedia agent for GDP/economic data
2. Implement hybrid mode for combining sources
3. Add confidence scoring for multi-source answers

**Estimated Impact**: +1 question (Q59)

## Code Changes Needed

### 1. Update db_agent.py SQL Prompt
```python
# Add to _generate_sql() prompt:
"""
Advanced Calculations:

- CAGR (Compound Annual Growth Rate):
  Formula: ((end/start)^(1/years) - 1) * 100
  SELECT (POWER(end_val / NULLIF(start_val, 0), 1.0/years) - 1) * 100

- Percentage Change:
  Formula: ((new - old) / old) * 100
  SELECT ((new_val - old_val) / NULLIF(old_val, 0)) * 100

- Complex Aggregations with CTEs:
  WITH base AS (SELECT SUM(x) as total FROM ...)
  SELECT total * factor FROM base
"""
```

### 2. Add Query Validator
```python
class SQLValidator:
    def validate_query(self, sql: str, schemas: dict) -> tuple[bool, str]:
        """Validate SQL before execution."""
        # Check column names exist
        # Check table names exist
        # Validate year ranges
        # Return (is_valid, error_message)
```

### 3. Implement Wikipedia Agent
```python
class WikipediaAgent(BaseAgent):
    async def process(self, question: str) -> AgentResponse:
        # Search Wikipedia for GDP/economic data
        # Parse infoboxes and tables
        # Return structured response
```

### 4. Implement Hybrid Mode
```python
# In agent.py
if routing.route_type == "hybrid":
    db_response = await self.db_agent.process(question)
    wiki_response = await self.wikipedia_agent.process(question)
    combined = self.aggregator.aggregate_hybrid([db_response, wiki_response])
```

## Expected Final Accuracy

With all Phase 3 improvements:
- Current: 36/43 (83.7%)
- After Priority 1: 39-40/43 (90-93%)
- After Priority 2: 40-42/43 (93-98%)
- After Priority 3: 42-43/43 (98-100%)

## Testing Strategy

1. **Unit Tests**: Add tests for each new SQL pattern
2. **Integration Tests**: Test CAGR, percentage change calculations
3. **Regression Tests**: Ensure Phase 1 fixes remain intact
4. **Incremental Deployment**: Apply one priority at a time, validate results

## Notes

- Phase 1 established solid foundation (81.4% → 83.7% with Q66 fix)
- Phase 2 showed that adding too many patterns can confuse SQL generation
- Phase 3 should add features incrementally with careful testing
- Focus on explicit, unambiguous patterns rather than general guidance
