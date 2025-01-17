The program loads the UNSW_NB15 training set dataset, preprocesses it, trains various classifiers and evaluate them.

Error Messages
Dataset Loaded. Shape: (82332, 45)
Traceback (most recent call last):
  File "/Users/apple/Desktop/msc_project/unwclassifier.py", line 28, in <module>
    X = scaler.fit_transform(X)
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/sklearn/utils/_set_output.py", line 319, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/sklearn/base.py", line 918, in fit_transform
    return self.fit(X, **fit_params).transform(X)
           ~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/sklearn/preprocessing/_data.py", line 894, in fit
    return self.partial_fit(X, y, sample_weight)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/sklearn/base.py", line 1389, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/sklearn/preprocessing/_data.py", line 930, in partial_fit
    X = validate_data(
        self,
    ...<4 lines>...
        reset=first_call,
    )
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/sklearn/utils/validation.py", line 2944, in validate_data
    out = check_array(X, input_name="X", **check_params)
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/sklearn/utils/validation.py", line 1055, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/sklearn/utils/_array_api.py", line 839, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)
  File "/Users/apple/Desktop/msc_project/env/lib/python3.13/site-packages/pandas/core/generic.py", line 2153, in __array__
    arr = np.asarray(values, dtype=dtype)
ValueError: could not convert string to float: 'udp'
