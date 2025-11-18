# Refactoring Summary - Version 0.3.1

## Changes Made

### Function Name Simplification

The Welch's t-test functions have been renamed for simplicity and ease of use:

| Old Name                   | New Name | Description                    |
|----------------------------|----------|--------------------------------|
| `welch_ttest_one_sample`   | `one_t`  | One-sample Welch's t-test      |
| `welch_ttest_two_sample`   | `two_t`  | Two-sample Welch's t-test      |

### Version Update

- **Previous version:** 0.3.0
- **New version:** 0.3.1

### Files Updated

1. **`src/quantpolars/ttest.py`**
   - Renamed `welch_ttest_one_sample` → `one_t`
   - Renamed `welch_ttest_two_sample` → `two_t`

2. **`src/quantpolars/__init__.py`**
   - Updated exports to use new function names
   - Bumped version to 0.3.1

3. **`pyproject.toml`**
   - Updated version to 0.3.1

4. **`tests/test_ttest.py`**
   - Updated all 27 test cases to use new function names
   - All tests pass ✅

5. **`demo_ttest.py`**
   - Updated all examples to use new function names
   - Demo runs successfully ✅

6. **`TTEST_DOCUMENTATION.md`**
   - Updated all documentation with new function names
   - Updated all code examples

## Usage Examples

### Before (v0.3.0)

```python
from quantpolars import welch_ttest_one_sample, welch_ttest_two_sample

# One-sample test
result = welch_ttest_one_sample(df, column="values", mu=0)

# Two-sample test
result = welch_ttest_two_sample(df, column1="col1", column2="col2")
```

### After (v0.3.1)

```python
from quantpolars import one_t, two_t

# One-sample test
result = one_t(df, column="values", mu=0)

# Two-sample test
result = two_t(df, column1="col1", column2="col2")
```

## Benefits

1. **Shorter function names** - Easier to type and remember
2. **Cleaner code** - More concise and readable
3. **Consistent with package philosophy** - Similar to existing `sm()` function
4. **Backward compatibility note** - This is a breaking change for v0.3.0 users

## Testing Status

- ✅ All 27 unit tests pass
- ✅ Demo script runs successfully
- ✅ Documentation updated
- ✅ Package builds and installs correctly

## Migration Guide

If you're upgrading from v0.3.0, simply replace:
- `welch_ttest_one_sample` with `one_t`
- `welch_ttest_two_sample` with `two_t`

All parameters and return values remain unchanged.
