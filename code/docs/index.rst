Myocardial Blood Flow Documentation
====================================

This documentation provides detailed information about the Myocardial Blood Flow analysis package,
which processes DICOM files and masks to compute myocardial blood flow parameters.

Overview
--------

The package consists of several modules that work together to:

- Load and process DICOM image files
- Extract myocardial regions using masks
- Compute blood flow parameters using mathematical models
- Save results and generate visualizations

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   class_diagrams

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/data_loader
   api/compute_quantity
   api/myocardial_blood_flow
   api/logging_config
   api/save_data_manager

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`