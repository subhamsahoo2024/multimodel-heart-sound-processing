"use client";

import React, { useState, useRef, useCallback } from "react";
import {
  Upload,
  Activity,
  Heart,
  AlertCircle,
  CheckCircle,
  Loader2,
  Play,
  Pause,
  FileText,
  Download,
  X,
  Eye,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Tabs } from "@/components/ui/Tabs";
import { SignalChart } from "@/components/charts/SignalChart";
import { SpectrogramViewer } from "@/components/charts/SpectrogramViewer";
import { RiskBarChart } from "@/components/charts/RiskBarChart";
import { uploadFiles, PredictionResponse } from "@/services/api";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";

export default function Home() {
  // State Management
  const [ecgFile, setEcgFile] = useState<File | null>(null);
  const [pcgFile, setPcgFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<number>(0);

  // Audio player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [pcgAudioUrl, setPcgAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // CSV Data modal state
  const [showCsvModal, setShowCsvModal] = useState(false);
  const [csvData, setCsvData] = useState<string[][] | null>(null);
  const [csvLoading, setCsvLoading] = useState(false);

  // Report generation state
  const [reportLoading, setReportLoading] = useState(false);

  // Refs for capturing charts
  const ecgChartRef = useRef<HTMLDivElement>(null);
  const pcgChartRef = useRef<HTMLDivElement>(null);
  const spectrogramRef = useRef<HTMLDivElement>(null);
  const riskChartRef = useRef<HTMLDivElement>(null);

  // Handle file selection
  const handleEcgChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setEcgFile(e.target.files[0]);
      setError(null);
      setCsvData(null); // Reset CSV data when new file is selected
    }
  };

  const handlePcgChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setPcgFile(file);
      setError(null);

      // Create audio URL for playback
      if (pcgAudioUrl) {
        URL.revokeObjectURL(pcgAudioUrl);
      }
      const audioUrl = URL.createObjectURL(file);
      setPcgAudioUrl(audioUrl);
      setIsPlaying(false);
    }
  };

  // Audio playback controls
  const toggleAudioPlayback = () => {
    if (!audioRef.current || !pcgAudioUrl) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  // Parse and display ECG CSV data
  const handleViewCsvData = async () => {
    if (!ecgFile) return;

    setCsvLoading(true);
    try {
      const text = await ecgFile.text();
      const rows = text
        .trim()
        .split("\n")
        .map((row) => row.split(","));
      setCsvData(rows);
      setShowCsvModal(true);
    } catch (err) {
      setError("Failed to parse CSV file");
    } finally {
      setCsvLoading(false);
    }
  };

  // Generate and download PDF report
  const generateReport = useCallback(async () => {
    if (!result) return;

    setReportLoading(true);
    try {
      const pdf = new jsPDF("p", "mm", "a4");
      const pageWidth = pdf.internal.pageSize.getWidth();
      const margin = 15;
      let yPos = 20;

      // Helper function to add section header
      const addSectionHeader = (text: string) => {
        pdf.setFontSize(14);
        pdf.setTextColor(16, 185, 129); // Emerald color
        pdf.text(text, margin, yPos);
        yPos += 8;
        pdf.setTextColor(0, 0, 0);
        pdf.setFontSize(11);
      };

      // Helper function to add text
      const addText = (label: string, value: string, indent: number = 0) => {
        pdf.setFont("helvetica", "bold");
        pdf.text(label + ": ", margin + indent, yPos);
        const labelWidth = pdf.getTextWidth(label + ": ");
        pdf.setFont("helvetica", "normal");
        pdf.text(value, margin + indent + labelWidth, yPos);
        yPos += 6;
      };

      // Title
      pdf.setFontSize(22);
      pdf.setTextColor(16, 185, 129);
      pdf.text("SmartHeart Analysis Report", pageWidth / 2, yPos, {
        align: "center",
      });
      yPos += 10;

      // Subtitle
      pdf.setFontSize(10);
      pdf.setTextColor(100, 100, 100);
      pdf.text(
        "Bimodal Deep Learning for ECG and PCG-Based Cardiac Monitoring",
        pageWidth / 2,
        yPos,
        { align: "center" },
      );
      yPos += 15;

      // Date and Time
      pdf.setFontSize(10);
      pdf.setTextColor(0, 0, 0);
      const now = new Date();
      pdf.text(
        `Report Generated: ${now.toLocaleDateString()} at ${now.toLocaleTimeString()}`,
        margin,
        yPos,
      );
      yPos += 15;

      // Divider line
      pdf.setDrawColor(200, 200, 200);
      pdf.line(margin, yPos - 5, pageWidth - margin, yPos - 5);

      // Risk Assessment Results
      addSectionHeader("Risk Assessment Results");
      yPos += 2;

      // ECG Risk
      const ecgRiskText =
        result.ecg_risk !== null
          ? `${(result.ecg_risk * 100).toFixed(1)}% (${result.ecg_risk > 0.5 ? "HIGH RISK" : "LOW RISK"})`
          : "Not Available";
      addText("ECG Risk", ecgRiskText);

      // PCG Risk
      const pcgRiskText =
        result.pcg_risk !== null
          ? `${(result.pcg_risk * 100).toFixed(1)}% (${result.pcg_risk > 0.5 ? "HIGH RISK" : "LOW RISK"})`
          : "Not Available";
      addText("PCG Risk", pcgRiskText);

      // Combined Risk
      const combinedRiskText =
        result.combined_risk !== null
          ? `${(result.combined_risk * 100).toFixed(1)}% (${result.combined_risk > 0.5 ? "HIGH RISK" : "LOW RISK"})`
          : "Not Available";
      addText("Combined Risk", combinedRiskText);

      yPos += 10;

      // Files Analyzed
      addSectionHeader("Files Analyzed");
      yPos += 2;
      if (ecgFile) {
        addText("ECG File", ecgFile.name);
      }
      if (pcgFile) {
        addText("PCG File", pcgFile.name);
      }

      yPos += 10;

      // Capture and add Risk Bar Chart
      if (riskChartRef.current) {
        try {
          const canvas = await html2canvas(riskChartRef.current, {
            backgroundColor: "#111827",
            scale: 2,
          });
          const imgData = canvas.toDataURL("image/png");
          const imgWidth = pageWidth - 2 * margin;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;

          // Check if we need a new page
          if (yPos + imgHeight > 280) {
            pdf.addPage();
            yPos = 20;
          }

          addSectionHeader("Risk Assessment Overview");
          pdf.addImage(imgData, "PNG", margin, yPos, imgWidth, imgHeight);
          yPos += imgHeight + 10;
        } catch (err) {
          console.error("Failed to capture risk chart:", err);
        }
      }

      // Capture ECG Chart if available
      if (result.ecg_plot_data && ecgChartRef.current) {
        try {
          // Switch to ECG tab temporarily for capture
          const canvas = await html2canvas(ecgChartRef.current, {
            backgroundColor: "#111827",
            scale: 2,
          });
          const imgData = canvas.toDataURL("image/png");
          const imgWidth = pageWidth - 2 * margin;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;

          // Check if we need a new page
          if (yPos + imgHeight > 280) {
            pdf.addPage();
            yPos = 20;
          }

          addSectionHeader("ECG Signal Visualization");
          pdf.addImage(imgData, "PNG", margin, yPos, imgWidth, imgHeight);
          yPos += imgHeight + 10;
        } catch (err) {
          console.error("Failed to capture ECG chart:", err);
        }
      }

      // Capture PCG Chart if available
      if (result.pcg_waveform_data && pcgChartRef.current) {
        try {
          const canvas = await html2canvas(pcgChartRef.current, {
            backgroundColor: "#111827",
            scale: 2,
          });
          const imgData = canvas.toDataURL("image/png");
          const imgWidth = pageWidth - 2 * margin;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;

          // Check if we need a new page
          if (yPos + imgHeight > 280) {
            pdf.addPage();
            yPos = 20;
          }

          addSectionHeader("PCG Waveform Visualization");
          pdf.addImage(imgData, "PNG", margin, yPos, imgWidth, imgHeight);
          yPos += imgHeight + 10;
        } catch (err) {
          console.error("Failed to capture PCG chart:", err);
        }
      }

      // Add Spectrogram if available
      if (result.pcg_spectrogram) {
        try {
          const imgWidth = pageWidth - 2 * margin;
          const imgHeight = 60; // Fixed height for spectrogram

          // Check if we need a new page
          if (yPos + imgHeight > 280) {
            pdf.addPage();
            yPos = 20;
          }

          addSectionHeader("PCG Spectrogram");
          pdf.addImage(
            result.pcg_spectrogram,
            "PNG",
            margin,
            yPos,
            imgWidth,
            imgHeight,
          );
          yPos += imgHeight + 10;
        } catch (err) {
          console.error("Failed to add spectrogram:", err);
        }
      }

      // Footer
      const pageCount = pdf.getNumberOfPages();
      for (let i = 1; i <= pageCount; i++) {
        pdf.setPage(i);
        pdf.setFontSize(8);
        pdf.setTextColor(150, 150, 150);
        pdf.text(
          "SmartHeart - AI-powered multimodal cardiovascular risk assessment system | For research and educational purposes only",
          pageWidth / 2,
          290,
          { align: "center" },
        );
        pdf.text(`Page ${i} of ${pageCount}`, pageWidth - margin, 290, {
          align: "right",
        });
      }

      // Download the PDF
      pdf.save(`SmartHeart_Report_${now.toISOString().split("T")[0]}.pdf`);
    } catch (err) {
      console.error("Report generation error:", err);
      setError("Failed to generate report. Please try again.");
    } finally {
      setReportLoading(false);
    }
  }, [result, ecgFile, pcgFile]);

  // Handle analysis
  const handleAnalyze = async () => {
    if (!ecgFile && !pcgFile) {
      setError("Please upload at least one file (ECG or PCG)");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await uploadFiles(ecgFile, pcgFile);
      setResult(response);
      setActiveTab(0); // Reset to first tab
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An error occurred during analysis",
      );
      console.error("Analysis error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Risk color helper
  const getRiskColor = (risk: number | null): string => {
    if (risk === null) return "text-gray-400";
    return risk > 0.5 ? "text-red-400" : "text-emerald-400";
  };

  const getRiskLabel = (risk: number | null): string => {
    if (risk === null) return "N/A";
    return risk > 0.5 ? "HIGH RISK" : "LOW RISK";
  };

  // Format ECG data for chart
  const getEcgChartData = () => {
    if (!result?.ecg_plot_data) return [];
    return result.ecg_plot_data.map(([x, y]) => ({ x, y }));
  };

  // Format PCG waveform data for chart
  const getPcgChartData = () => {
    if (!result?.pcg_waveform_data) return [];
    return result.pcg_waveform_data.map((y, x) => ({ x, y }));
  };

  return (
    <div className="min-h-screen p-6 md:p-8 lg:p-12">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-3">
          <div className="flex items-center justify-center gap-3">
            <Heart className="w-10 h-10 text-emerald-500 animate-pulse" />
            <h1 className="text-4xl md:text-5xl font-bold bg-linear-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent">
              SmartHeart
            </h1>
          </div>
          <p className="text-gray-400 text-lg">
            Bimodal Deep Learning for ECG and PCG–Based Cardiac Monitoring
          </p>
        </div>

        {/* Upload Section */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* ECG Upload */}
          <Card
            title="ECG Signal Upload"
            className="hover:border-blue-500/50 transition-colors"
          >
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Activity className="w-4 h-4" />
                <span>Electrocardiogram (.csv format)</span>
              </div>

              <label className="block">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleEcgChange}
                  className="hidden"
                  disabled={loading}
                />
                <div className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-gray-800/30 transition-all">
                  <Upload className="w-8 h-8 mx-auto mb-3 text-gray-500" />
                  <p className="text-sm text-gray-400">
                    {ecgFile ? (
                      <span className="text-blue-400 font-medium">
                        {ecgFile.name}
                      </span>
                    ) : (
                      "Click to upload ECG file"
                    )}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">
                    187 sample points required
                  </p>
                </div>
              </label>

              {/* View ECG CSV Data Button */}
              {ecgFile && (
                <button
                  onClick={handleViewCsvData}
                  disabled={csvLoading}
                  className="w-full mt-3 px-4 py-2.5 bg-blue-500/20 border border-blue-500/50 rounded-lg
                           text-blue-400 text-sm font-medium flex items-center justify-center gap-2
                           hover:bg-blue-500/30 hover:border-blue-500 transition-all disabled:opacity-50"
                >
                  {csvLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                  View ECG Data
                </button>
              )}
            </div>
          </Card>

          {/* PCG Upload */}
          <Card
            title="PCG Audio Upload"
            className="hover:border-emerald-500/50 transition-colors"
          >
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Heart className="w-4 h-4" />
                <span>Phonocardiogram (.wav format)</span>
              </div>

              <label className="block">
                <input
                  type="file"
                  accept=".wav"
                  onChange={handlePcgChange}
                  className="hidden"
                  disabled={loading}
                />
                <div className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-emerald-500 hover:bg-gray-800/30 transition-all">
                  <Upload className="w-8 h-8 mx-auto mb-3 text-gray-500" />
                  <p className="text-sm text-gray-400">
                    {pcgFile ? (
                      <span className="text-emerald-400 font-medium">
                        {pcgFile.name}
                      </span>
                    ) : (
                      "Click to upload PCG file"
                    )}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">
                    5-second heart sound recording
                  </p>
                </div>
              </label>

              {/* Audio Player for PCG */}
              {pcgFile && pcgAudioUrl && (
                <div className="mt-3">
                  <audio
                    ref={audioRef}
                    src={pcgAudioUrl}
                    onEnded={handleAudioEnded}
                    className="hidden"
                  />
                  <button
                    onClick={toggleAudioPlayback}
                    className="w-full px-4 py-2.5 bg-emerald-500/20 border border-emerald-500/50 rounded-lg
                             text-emerald-400 text-sm font-medium flex items-center justify-center gap-2
                             hover:bg-emerald-500/30 hover:border-emerald-500 transition-all"
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="w-4 h-4" />
                        Pause Audio
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Play PCG Audio
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* Analyze Button */}
        <div className="flex flex-col items-center gap-4">
          <button
            onClick={handleAnalyze}
            disabled={loading || (!ecgFile && !pcgFile)}
            className="px-8 py-4 bg-linear-to-r from-emerald-500 to-blue-500 rounded-lg font-semibold text-white text-lg
                     hover:from-emerald-600 hover:to-blue-600 disabled:from-gray-700 disabled:to-gray-700 
                     disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-emerald-500/25
                     flex items-center gap-3"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Activity className="w-5 h-5" />
                Analyze Heart Data
              </>
            )}
          </button>

          {error && (
            <div className="flex items-center gap-2 text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* Download Report Button - shows after results */}
          {result && (
            <button
              onClick={generateReport}
              disabled={reportLoading}
              className="px-6 py-3 bg-gray-800 border border-gray-700 rounded-lg font-medium text-gray-300
                       hover:bg-gray-750 hover:border-gray-600 transition-all duration-300
                       flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {reportLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating Report...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Download Report (PDF)
                </>
              )}
            </button>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="space-y-8 animate-in fade-in duration-500">
            {/* Risk Cards */}
            <div className="grid md:grid-cols-3 gap-6">
              {/* ECG Risk */}
              <Card className="text-center">
                <div className="space-y-3">
                  <div className="flex items-center justify-center gap-2 text-blue-400">
                    <Activity className="w-5 h-5" />
                    <h3 className="font-semibold">ECG Risk</h3>
                  </div>
                  <div
                    className={`text-4xl font-bold ${getRiskColor(result.ecg_risk)}`}
                  >
                    {result.ecg_risk !== null
                      ? (result.ecg_risk * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                  <div
                    className={`text-sm font-medium ${getRiskColor(result.ecg_risk)}`}
                  >
                    {getRiskLabel(result.ecg_risk)}
                  </div>
                </div>
              </Card>

              {/* PCG Risk */}
              <Card className="text-center">
                <div className="space-y-3">
                  <div className="flex items-center justify-center gap-2 text-emerald-400">
                    <Heart className="w-5 h-5" />
                    <h3 className="font-semibold">PCG Risk</h3>
                  </div>
                  <div
                    className={`text-4xl font-bold ${getRiskColor(result.pcg_risk)}`}
                  >
                    {result.pcg_risk !== null
                      ? (result.pcg_risk * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                  <div
                    className={`text-sm font-medium ${getRiskColor(result.pcg_risk)}`}
                  >
                    {getRiskLabel(result.pcg_risk)}
                  </div>
                </div>
              </Card>

              {/* Combined Risk */}
              <Card className="text-center border-emerald-500/30 bg-emerald-500/5">
                <div className="space-y-3">
                  <div className="flex items-center justify-center gap-2 text-emerald-400">
                    {result.combined_risk !== null &&
                    result.combined_risk > 0.5 ? (
                      <AlertCircle className="w-5 h-5" />
                    ) : (
                      <CheckCircle className="w-5 h-5" />
                    )}
                    <h3 className="font-semibold">Combined Risk</h3>
                  </div>
                  <div
                    className={`text-5xl font-bold ${getRiskColor(result.combined_risk)}`}
                  >
                    {result.combined_risk !== null
                      ? (result.combined_risk * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                  <div
                    className={`text-sm font-medium ${getRiskColor(result.combined_risk)}`}
                  >
                    {getRiskLabel(result.combined_risk)}
                  </div>
                </div>
              </Card>
            </div>

            {/* Risk Comparison Bar Chart */}
            <Card title="Risk Assessment Overview">
              <div className="h-80" ref={riskChartRef}>
                <RiskBarChart
                  ecgRisk={result.ecg_risk}
                  pcgRisk={result.pcg_risk}
                  combinedRisk={result.combined_risk}
                />
              </div>
            </Card>

            {/* Visuals Container */}
            <Card title="Signal Visualization">
              <div className="space-y-6">
                {/* Tabs */}
                <Tabs
                  labels={["ECG Signal", "PCG Waveform", "Spectrogram"]}
                  onChange={setActiveTab}
                  defaultActiveIndex={0}
                />

                {/* Chart Container */}
                <div className="h-100 bg-gray-950/50 rounded-lg p-4 border border-gray-800">
                  {activeTab === 0 && result.ecg_plot_data && (
                    <div ref={ecgChartRef} className="h-full">
                      <SignalChart
                        data={getEcgChartData()}
                        heatmapData={result.ecg_heatmap || undefined}
                        riskScore={
                          result.ecg_risk !== null
                            ? result.ecg_risk * 100
                            : undefined
                        }
                        color="#3b82f6"
                        title="ECG Waveform with Risk Heatmap"
                        xLabel="Sample"
                        yLabel="Normalized Amplitude"
                      />
                    </div>
                  )}

                  {activeTab === 1 && result.pcg_waveform_data && (
                    <div ref={pcgChartRef} className="h-full">
                      <SignalChart
                        data={getPcgChartData()}
                        heatmapData={result.pcg_heatmap || undefined}
                        riskScore={
                          result.pcg_risk !== null
                            ? result.pcg_risk * 100
                            : undefined
                        }
                        color="#10b981"
                        title="PCG Audio Waveform with Risk Heatmap"
                        xLabel="Sample"
                        yLabel="Amplitude"
                      />
                    </div>
                  )}

                  {activeTab === 2 && (
                    <div ref={spectrogramRef} className="h-full">
                      <SpectrogramViewer
                        base64Image={result.pcg_spectrogram}
                        title="PCG Frequency Spectrogram"
                      />
                    </div>
                  )}

                  {/* Empty state for missing data */}
                  {((activeTab === 0 && !result.ecg_plot_data) ||
                    (activeTab === 1 && !result.pcg_waveform_data) ||
                    (activeTab === 2 && !result.pcg_spectrogram)) && (
                    <div className="flex items-center justify-center h-full">
                      <div className="text-center text-gray-500">
                        <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>No data available for this visualization</p>
                        <p className="text-sm text-gray-600 mt-1">
                          Upload the corresponding file to view
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-sm text-gray-600 pt-8 border-t border-gray-800">
          <p>AI-powered multimodal cardiovascular risk assessment system</p>
          <p className="mt-3">For research and educational purposes only</p>
        </div>
      </div>

      {/* CSV Data Modal */}
      {showCsvModal && csvData && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-xl max-w-4xl w-full max-h-[80vh] flex flex-col shadow-2xl">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">
                  ECG CSV Data
                </h3>
                <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">
                  {csvData.length} rows × {csvData[0]?.length || 0} columns
                </span>
              </div>
              <button
                onClick={() => setShowCsvModal(false)}
                className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-400 hover:text-white" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="overflow-auto flex-1 p-4">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-800">
                  <tr>
                    <th className="px-3 py-2 text-left text-gray-400 font-medium border-b border-gray-700">
                      #
                    </th>
                    {csvData[0]?.map((_, colIndex) => (
                      <th
                        key={colIndex}
                        className="px-3 py-2 text-left text-gray-400 font-medium border-b border-gray-700"
                      >
                        Col {colIndex + 1}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {csvData.slice(0, 200).map((row, rowIndex) => (
                    <tr key={rowIndex} className="hover:bg-gray-800/50">
                      <td className="px-3 py-1.5 text-gray-500 border-b border-gray-800">
                        {rowIndex + 1}
                      </td>
                      {row.map((cell, cellIndex) => (
                        <td
                          key={cellIndex}
                          className="px-3 py-1.5 text-gray-300 border-b border-gray-800 font-mono text-xs"
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {csvData.length > 200 && (
                <p className="text-center text-gray-500 text-sm mt-4">
                  Showing first 200 rows of {csvData.length} total rows
                </p>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-gray-700 flex justify-end">
              <button
                onClick={() => setShowCsvModal(false)}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 text-sm transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
