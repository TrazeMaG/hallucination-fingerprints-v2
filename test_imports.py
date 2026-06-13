python -c "
import warnings
warnings.filterwarnings('error')
try:
    from transformer_lens import HookedTransformer
    print('ok')
except Exception as e:
    print(type(e).__name__, e)
"