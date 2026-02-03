import axios, { AxiosError } from "axios";

// Backend API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * TypeScript interface for the backend prediction response
 */
export interface PredictionResponse {
  ecg_risk: number | null;
  pcg_risk: number | null;
  combined_risk: number | null;
  ecg_plot_data: [number, number][] | null; // Array of [x, y] tuples
  pcg_waveform_data: number[] | null; // Array of amplitude values
  pcg_spectrogram: string | null; // Base64 encoded image string
  ecg_heatmap: number[] | null; // Grad-CAM heatmap for ECG (0-1 values)
  pcg_heatmap: number[] | null; // Grad-CAM heatmap for PCG (0-1 values)
}

/**
 * API Error response structure
 */
export interface ApiError {
  detail: string;
  status?: number;
}

/**
 * Upload ECG and/or PCG files to the FastAPI backend for analysis
 *
 * @param ecgFile - ECG CSV file (optional)
 * @param pcgFile - PCG WAV audio file (optional)
 * @returns Promise with prediction results
 * @throws ApiError if request fails
 */
export async function uploadFiles(
  ecgFile: File | null,
  pcgFile: File | null,
): Promise<PredictionResponse> {
  try {
    // Validate that at least one file is provided
    if (!ecgFile && !pcgFile) {
      throw new Error("At least one file (ECG or PCG) must be provided");
    }

    // Create FormData and append files only if they exist
    const formData = new FormData();

    if (ecgFile) {
      formData.append("ecg_file", ecgFile);
      console.log(
        `Uploading ECG file: ${ecgFile.name} (${ecgFile.size} bytes)`,
      );
    }

    if (pcgFile) {
      formData.append("pcg_file", pcgFile);
      console.log(
        `Uploading PCG file: ${pcgFile.name} (${pcgFile.size} bytes)`,
      );
    }

    // Send POST request to backend
    const response = await axios.post<PredictionResponse>(
      `${API_BASE_URL}/predict`,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 300000, // 5 minutes timeout
      },
    );

    console.log("Prediction successful:", response.data);
    return response.data;
  } catch (error) {
    // Handle Axios errors
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ApiError>;

      if (axiosError.response) {
        // Server responded with error status
        const errorMessage =
          axiosError.response.data?.detail || "Server error occurred";
        console.error("API Error:", {
          status: axiosError.response.status,
          message: errorMessage,
        });

        throw new Error(
          `${errorMessage} (Status: ${axiosError.response.status})`,
        );
      } else if (axiosError.request) {
        // Request made but no response received
        console.error("Network Error: No response from server");
        throw new Error(
          "Network error: Unable to reach the server. Please check if the backend is running.",
        );
      }
    }

    // Handle other errors
    if (error instanceof Error) {
      console.error("Upload Error:", error.message);
      throw error;
    }

    // Unknown error
    console.error("Unknown error:", error);
    throw new Error("An unexpected error occurred during file upload");
  }
}

/**
 * Check backend health status
 *
 * @returns Promise<boolean> - true if backend is healthy
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`, {
      timeout: 5000,
    });
    return response.status === 200;
  } catch (error) {
    console.error("Health check failed:", error);
    return false;
  }
}

/**
 * Get backend API information
 *
 * @returns Promise with API info
 */
export async function getApiInfo(): Promise<any> {
  try {
    const response = await axios.get(`${API_BASE_URL}/`);
    return response.data;
  } catch (error) {
    console.error("Failed to fetch API info:", error);
    throw error;
  }
}

export default {
  uploadFiles,
  checkHealth,
  getApiInfo,
};
