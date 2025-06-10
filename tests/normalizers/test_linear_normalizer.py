import pytest
import torch

from my_normalizers.linear import LinearNormalizer


@pytest.fixture
def sample_dict_data():
    """Provides a sample dictionary of data for testing."""
    return {
        "action": torch.randn(100, 7) * 2 + 5,  # mean=5, std=2
        "agent_pos": torch.linspace(-10, 10, steps=100)
        .unsqueeze(-1)
        .repeat(1, 2),  # range [-10, 10]
    }


@pytest.fixture
def sample_tensor_data():
    """Provides a single sample tensor for testing."""
    return torch.randn(100, 3) * 5 - 2  # mean=-2, std=5


class TestLinearNormalizer:
    """
    A suite of unit tests for the LinearNormalizer class.
    """

    def test_initialization(self):
        """Tests that the normalizer initializes in an empty state."""
        normalizer = LinearNormalizer()
        assert not normalizer.params

    def test_unfitted_error(self, sample_dict_data):
        """Tests that using the normalizer before fitting raises a RuntimeError."""
        normalizer = LinearNormalizer()
        with pytest.raises(RuntimeError, match="Normalizer has not been fitted"):
            normalizer.normalize(sample_dict_data)
        with pytest.raises(RuntimeError, match="Normalizer has not been fitted"):
            normalizer.unnormalize(sample_dict_data)

    # --- Gaussian Mode Tests ---

    def test_fit_and_transform_gaussian_dict(self, sample_dict_data):
        """Tests fitting and transformation with dictionary data in gaussian mode."""
        normalizer = LinearNormalizer()
        normalizer.fit(sample_dict_data, mode="gaussian")

        # Check if params were created for each key
        assert "action" in normalizer.params
        assert "agent_pos" in normalizer.params

        normalized_data = normalizer.normalize(sample_dict_data)

        # Check if the normalized data has approximately mean=0 and std=1
        print("Normalized action mean:", torch.mean(normalized_data["action"]))
        print("Normalized action std:", torch.std(normalized_data["action"]))

        torch.testing.assert_close(
            torch.mean(normalized_data["action"]),
            torch.tensor(0.0),
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            torch.std(normalized_data["action"]),
            torch.tensor(1.0),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_invertibility_gaussian(self, sample_dict_data):
        """Tests if unnormalize(normalize(x)) returns the original data in gaussian mode."""
        normalizer = LinearNormalizer()
        normalizer.fit(sample_dict_data, mode="gaussian")

        normalized_data = normalizer.normalize(sample_dict_data)
        unnormalized_data = normalizer.unnormalize(normalized_data)

        torch.testing.assert_close(
            unnormalized_data["action"], sample_dict_data["action"]
        )
        torch.testing.assert_close(
            unnormalized_data["agent_pos"], sample_dict_data["agent_pos"]
        )

    # --- Limits Mode Tests ---

    def test_fit_and_transform_limits_tensor(self, sample_tensor_data):
        """Tests fitting and transformation with a single tensor in limits mode."""
        normalizer = LinearNormalizer()
        output_min, output_max = -1.0, 1.0
        normalizer.fit(
            sample_tensor_data,
            mode="limits",
            output_min=output_min,
            output_max=output_max,
        )

        assert "_default" in normalizer.params

        normalized_data = normalizer.normalize(sample_tensor_data)

        assert torch.all(normalized_data >= output_min)
        assert torch.all(normalized_data <= output_max)
        # Check if the min and max of the normalized data match the limits
        torch.testing.assert_close(torch.min(normalized_data), torch.tensor(output_min))
        torch.testing.assert_close(torch.max(normalized_data), torch.tensor(output_max))

    def test_invertibility_limits(self, sample_tensor_data):
        """Tests if unnormalize(normalize(x)) returns the original data in limits mode."""
        normalizer = LinearNormalizer()
        normalizer.fit(sample_tensor_data, mode="limits")

        normalized_data = normalizer.normalize(sample_tensor_data)
        unnormalized_data = normalizer.unnormalize(normalized_data)

        torch.testing.assert_close(unnormalized_data, sample_tensor_data)

    # --- Edge Cases and Error Handling ---

    def test_constant_feature_handling(self):
        """Tests that constant features (zero std/range) don't cause division by zero."""
        data = {"const": torch.ones(100, 2)}
        normalizer = LinearNormalizer()

        # Gaussian mode
        normalizer.fit(data, mode="gaussian")
        normalized = normalizer.normalize(data)
        # Without range_eps, scale would be inf. With it, it should be a large number but not inf.
        # The normalized output should be finite (likely 0 if offset is fit).
        assert torch.all(torch.isfinite(normalized["const"]))

        # Limits mode
        normalizer.fit(data, mode="limits")
        normalized = normalizer.normalize(data)
        assert torch.all(torch.isfinite(normalized["const"]))

    def test_key_error(self, sample_dict_data):
        """Tests that a KeyError is raised for unseen keys."""
        normalizer = LinearNormalizer()
        normalizer.fit(sample_dict_data)

        bad_data = {"unseen_key": torch.randn(10, 2)}
        with pytest.raises(
            KeyError, match="No normalization parameters found for key: 'unseen_key'"
        ):
            normalizer.normalize(bad_data)

    def test_mismatched_fit_and_transform_error(
        self, sample_dict_data, sample_tensor_data
    ):
        """Tests that a RuntimeError is raised if fit/transform types mismatch."""
        normalizer = LinearNormalizer()
        # Fit on a dict
        normalizer.fit(sample_dict_data)
        # Try to transform a single tensor
        with pytest.raises(RuntimeError, match="was fitted on a dictionary"):
            normalizer.normalize(sample_tensor_data)

    # --- State and Device Management ---

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_portability(self, sample_dict_data):
        """Tests that the normalizer and its parameters move to CUDA correctly."""
        normalizer = LinearNormalizer()
        normalizer.fit(sample_dict_data)

        # Move normalizer to GPU
        cuda_normalizer = normalizer.to("cuda")
        assert cuda_normalizer.params["action"]["scale"].device.type == "cuda"

        # Move data to GPU
        cuda_data = {k: v.to("cuda") for k, v in sample_dict_data.items()}

        # Perform normalization on GPU
        normalized_data_cuda = cuda_normalizer.normalize(cuda_data)
        assert normalized_data_cuda["action"].device.type == "cuda"

        # Unnormalize on GPU
        unnormalized_data_cuda = cuda_normalizer.unnormalize(normalized_data_cuda)
        assert unnormalized_data_cuda["action"].device.type == "cuda"

        # Check for correctness
        torch.testing.assert_close(
            unnormalized_data_cuda["action"].cpu(), sample_dict_data["action"]
        )

    # --- Test SingleKeyNormalizer Wrapper ---

    def test_single_key_normalizer_wrapper(self, sample_dict_data):
        """Tests the __getitem__ wrapper for single-key normalization."""
        normalizer = LinearNormalizer()
        normalizer.fit(sample_dict_data, mode="gaussian")

        # Get a wrapper for the 'action' key
        action_normalizer = normalizer["action"]

        action_data = sample_dict_data["action"]

        # Normalize and check
        normalized_action = action_normalizer.normalize(action_data)
        torch.testing.assert_close(
            torch.mean(normalized_action), torch.tensor(0.0), atol=1e-5, rtol=1e-5
        )

        # Unnormalize and check for invertibility
        unnormalized_action = action_normalizer.unnormalize(normalized_action)
        torch.testing.assert_close(unnormalized_action, action_data)
