Class Diagrams
==============

This page contains visual representations of the class hierarchy and relationships in the Myocardial Blood Flow package.

Class Inheritance Hierarchy
---------------------------

.. code-block:: text

   Base Classes and Inheritance:
   ┌─────────────────┐
   │  ComputeQuantity │ ← Base class for all computations
   └─────────────────┘
            │
            │ (inherits)
            ▼
   ┌─────────────────┐
   │MyocardialBloodFlow│ ← Main analysis class
   └─────────────────┘

Class Relationships
-------------------

The following diagram shows how the main classes interact:

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐
   │   DataLoader    │    │ SaveDataManager │
   │                 │    │                 │
   │ • Loads DICOM   │    │ • Saves results │
   │ • Processes     │    │ • Creates PNGs  │
   │   images        │    │ • Generates     │
   │ • Handles masks │    │   videos        │
   └─────────────────┘    └─────────────────┘
            ▲                       ▲
            │                       │
            │ (uses)                │ (uses)
            │                       │
   ┌─────────────────┐               │
   │MyocardialBloodFlow│ ←────────────┘
   │                 │
   │ • Computes MBF  │
   │ • Analyzes      │
   │   perfusion     │
   │ • Uses Fermi    │
   │   functions     │
   └─────────────────┘
            │
            │ (uses)
            ▼
   ┌─────────────────┐
   │  setup_logger() │
   │                 │
   │ • Configures    │
   │   logging       │
   │ • Creates       │
   │   loggers       │
   └─────────────────┘

Detailed Class Descriptions
---------------------------

**ComputeQuantity** (Base Class)
   The foundation class that provides core functionality for processing DICOM data and masks.
   All computation classes inherit from this base class.

   Key methods:
   - ``__init__()``: Initializes with frames and masks
   - ``_pixel_time_series()``: Extracts time series data for regions

**MyocardialBloodFlow** (Main Analysis Class)
   Extends ComputeQuantity to perform myocardial blood flow analysis using mathematical models.

   Key methods:
   - ``arterial_input_function()``: Computes AIF from blood pool data
   - ``myocardium_time_series()``: Extracts myocardial pixel data
   - ``fermi_function()``: Implements the Fermi impulse response function
   - ``calculate_mbf()``: Main MBF calculation method

**DataLoader** (Utility Class)
   Handles loading and preprocessing of DICOM files and associated mask data.

   Key methods:
   - ``load_dicom_series()``: Loads DICOM image sequences
   - ``load_masks()``: Loads myocardial and blood pool masks
   - ``validate_data()``: Ensures data integrity

**SaveDataManager** (Utility Class)
   Manages all data saving and visualization operations.

   Key methods:
   - ``save_image()``: Saves processed images as PNG files
   - ``save_metadata()``: Stores analysis metadata as JSON
   - ``create_movie()``: Generates DICOM sequence videos

**setup_logger()** (Utility Function)
   Configures logging for the entire application with appropriate formatting and levels.

Data Flow
---------

.. code-block:: text

   DICOM Files + Masks → DataLoader → MyocardialBloodFlow → SaveDataManager → Results
                              ↓              ↓                        ↓
                       Validation    Analysis & Computation    PNGs/Videos/JSON
