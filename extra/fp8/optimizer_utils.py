"""
Optimizer utilities for FP8 training with automatic cache management.

Provides two approaches for managing FP8 weight cache invalidation:
1. FP8Optimizer - Automatic invalidation via optimizer wrapper
2. invalidate_all_fp8_caches() - Manual invalidation helper

Usage Examples:

    # Automatic invalidation
    from extra.fp8 import FP8Optimizer
    from tinygrad.nn.optim import SGD

    model = get_mlperf_bert_model()  # FP8_CACHED=1
    optimizer = FP8Optimizer(SGD(model.parameters(), lr=0.01), model)

    for batch in dataloader:
        output = model(batch)
        loss.backward()
        optimizer.step()  # Auto-invalidates FP8 caches
        optimizer.zero_grad()

    # Manual invalidation
    from extra.fp8 import invalidate_all_fp8_caches
    from tinygrad.nn.optim import SGD

    model = get_mlperf_bert_model()  # FP8_CACHED=1
    optimizer = SGD(model.parameters(), lr=0.01)

    for batch in dataloader:
        output = model(batch)
        loss.backward()
        optimizer.step()
        invalidate_all_fp8_caches(model)  # Manual invalidation
        optimizer.zero_grad()
"""


def invalidate_all_fp8_caches(model):
    """
    Manually invalidate all FP8 weight caches in a model.

    Recursively searches the model for FP8LinearCached layers and calls
    invalidate_cache() on each one. Use this after optimizer.step() when
    not using the FP8Optimizer wrapper.

    Args:
        model: The model containing FP8LinearCached layers. Can be any object
               with attributes (e.g., BERT model, custom model).

    Returns:
        None

    Examples:
        Basic usage:
        >>> from extra.fp8 import FP8LinearCached, invalidate_all_fp8_caches
        >>> model = MyModel()  # Contains FP8LinearCached layers
        >>> optimizer.step()
        >>> invalidate_all_fp8_caches(model)  # Invalidate all caches

        BERT training:
        >>> model = get_mlperf_bert_model()  # FP8_CACHED=1
        >>> for batch in dataloader:
        ...     output = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     invalidate_all_fp8_caches(model)  # Clear caches
        ...     optimizer.zero_grad()

    Note:
        - Safe to call even if model has no FP8LinearCached layers
        - Handles nested models and complex hierarchies
        - Ignores private attributes (starting with '_')
    """
    # Import here to avoid circular dependency
    try:
        from extra.fp8.fp8_linear_cached import FP8LinearCached
    except ImportError:
        # If import fails, FP8LinearCached not available - nothing to invalidate
        return

    def invalidate_recursive(obj, visited=None):
        """Recursively traverse object and invalidate FP8 caches"""
        if visited is None:
            visited = set()

        # Avoid infinite recursion on circular references
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Check if this object is a FP8LinearCached layer
        if isinstance(obj, FP8LinearCached):
            obj.invalidate_cache()

        # Only search in __dict__ directly, avoid triggering descriptors/properties
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                # Skip private attributes and non-objects
                if attr_name.startswith('_') or attr_value is None:
                    continue

                try:
                    # Only recurse on objects with __dict__ (not Tensors, primitives, etc.)
                    if hasattr(attr_value, '__dict__') and not isinstance(attr_value, type):
                        invalidate_recursive(attr_value, visited)
                except (AttributeError, TypeError, RecursionError):
                    # Some attributes may not be accessible or may raise errors
                    pass

    invalidate_recursive(model)


class FP8Optimizer:
    """
    Optimizer wrapper that automatically invalidates FP8 weight caches after step().

    Wraps any tinygrad optimizer and ensures cached quantized weights are invalidated
    when parameters are updated. This eliminates the need to manually call
    invalidate_cache() on each FP8LinearCached layer.

    The wrapper discovers all FP8LinearCached layers in the model during initialization
    and automatically invalidates their caches after each optimizer.step().

    Args:
        optimizer: The underlying optimizer (SGD, Adam, AdamW, LAMB, etc.)
        model: The model containing FP8LinearCached layers

    Attributes:
        optimizer: The wrapped optimizer instance
        model: The model being optimized
        _fp8_layers: List of discovered FP8LinearCached layers

    Methods:
        step(): Execute optimizer step and invalidate caches
        zero_grad(): Forward to underlying optimizer
        __getattr__(): Forward any other methods to underlying optimizer

    Examples:
        Basic usage:
        >>> from extra.fp8 import FP8Optimizer, FP8LinearCached
        >>> from tinygrad.nn.optim import SGD
        >>>
        >>> model = MyModel()  # Contains FP8LinearCached layers
        >>> base_optimizer = SGD(model.parameters(), lr=0.01)
        >>> optimizer = FP8Optimizer(base_optimizer, model)
        >>>
        >>> # Training loop
        >>> for batch in dataloader:
        ...     output = model(batch)
        ...     loss.backward()
        ...     optimizer.step()  # Auto-invalidates FP8 caches
        ...     optimizer.zero_grad()

        BERT training with gradient accumulation:
        >>> from extra.fp8 import FP8Optimizer
        >>> from tinygrad.nn.optim import Adam
        >>>
        >>> model = get_mlperf_bert_model()  # FP8_CACHED=1
        >>> optimizer = FP8Optimizer(Adam(model.parameters()), model)
        >>>
        >>> for batch in dataloader:
        ...     for micro_batch in split_batch(batch, accumulation_steps=4):
        ...         output = model(micro_batch)  # First: quantize, rest: use cache
        ...         loss.backward()
        ...     optimizer.step()  # Auto-invalidates caches
        ...     optimizer.zero_grad()

    Note:
        - Compatible with all tinygrad optimizers (SGD, Adam, AdamW, LAMB, etc.)
        - Discovers layers once during __init__ (static discovery)
        - Minimal overhead: just iterates FP8 layers after step()
        - Can be nested/wrapped with other optimizer wrappers
    """

    def __init__(self, optimizer, model):
        """
        Initialize FP8Optimizer wrapper.

        Args:
            optimizer: Underlying tinygrad optimizer
            model: Model containing FP8LinearCached layers
        """
        self.optimizer = optimizer
        self.model = model
        self._fp8_layers = []

        # Discover all FP8LinearCached layers in model
        self._discover_fp8_layers()

    def _discover_fp8_layers(self):
        """
        Recursively find all FP8LinearCached layers in the model.

        Traverses the model hierarchy and builds a list of FP8LinearCached
        layer references for efficient invalidation during step().
        """
        # Import here to avoid circular dependency
        try:
            from extra.fp8.fp8_linear_cached import FP8LinearCached
        except ImportError:
            # If import fails, no FP8LinearCached layers available
            return

        def find_layers(obj, visited=None):
            """Recursively search for FP8LinearCached layers"""
            if visited is None:
                visited = set()

            layers = []

            # Avoid infinite recursion
            obj_id = id(obj)
            if obj_id in visited:
                return layers
            visited.add(obj_id)

            # Check if this is a FP8LinearCached layer
            if isinstance(obj, FP8LinearCached):
                layers.append(obj)

            # Only search in __dict__ directly, avoid triggering descriptors/properties
            if hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    # Skip private attributes and non-objects
                    if attr_name.startswith('_') or attr_value is None:
                        continue

                    try:
                        # Only recurse on objects with __dict__ (not Tensors, primitives, etc.)
                        if hasattr(attr_value, '__dict__') and not isinstance(attr_value, type):
                            layers.extend(find_layers(attr_value, visited))
                    except (AttributeError, TypeError, RecursionError):
                        pass

            return layers

        self._fp8_layers = find_layers(self.model)

    def step(self):
        """
        Execute optimizer step and invalidate FP8 caches.

        Calls the underlying optimizer's step() method, then invalidates
        all cached weights in FP8LinearCached layers.

        Returns:
            Return value from underlying optimizer.step() (if any)
        """
        # Execute underlying optimizer step
        result = self.optimizer.step()

        # Invalidate all cached FP8 weights
        for layer in self._fp8_layers:
            layer.invalidate_cache()

        return result

    def zero_grad(self):
        """
        Zero gradients.

        Forwards to underlying optimizer's zero_grad() method.
        """
        return self.optimizer.zero_grad()

    def __getattr__(self, name):
        """
        Forward attribute access to underlying optimizer.

        Allows transparent access to optimizer methods and attributes
        (e.g., learning rate, parameter groups, etc.).

        Args:
            name: Attribute name

        Returns:
            Attribute from underlying optimizer

        Raises:
            AttributeError: If attribute doesn't exist on underlying optimizer
        """
        return getattr(self.optimizer, name)

    def __repr__(self):
        """String representation showing wrapped optimizer."""
        return f"FP8Optimizer({self.optimizer})"
