package supervised;
import java.util.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.*;
import java.text.DecimalFormat;
public class Backpropagation {
	
	public double[][] input = {{0.32,0.46,0.64,0.75,0.89,1.07,1.18,1.32,1.5,1.61,1.75,1.93,2.04,2.18,2.36,2.47,2.61,2.79,2.9},{0.49,0.63,0.81,0.92,1.06,1.24,1.35,1.49,1.67,1.78,1.92,2.1,2.21,2.35,2.53,2.64,2.78,2.96,3.07},{0.21,0.35,0.53,0.64,0.78,0.96,1.07,1.21,1.39,1.5,1.64,1.82,1.93,2.07,2.25,2.36,2.5,2.68,2.79},{0.49,0.63,0.81,0.92,1.06,1.24,1.35,1.49,1.67,1.78,1.92,2.1,2.21,2.35,2.53,2.64,2.78,2.96,3.07},{0.6,0.74,0.92,1.03,1.17,1.35,1.46,1.6,1.78,1.89,2.03,2.21,2.32,2.46,2.64,2.75,2.89,3.07,3.18},{0.62,0.76,0.94,1.05,1.19,1.37,1.48,1.62,1.8,1.91,2.05,2.23,2.34,2.48,2.66,2.77,2.91,3.09,3.2},{0.41,0.55,0.73,0.84,0.98,1.16,1.27,1.41,1.59,1.7,1.84,2.02,2.13,2.27,2.45,2.56,2.7,2.88,2.99},{0.51,0.65,0.83,0.94,1.08,1.26,1.37,1.51,1.69,1.8,1.94,2.12,2.23,2.37,2.55,2.66,2.8,2.98,3.09},{0.14,0.28,0.46,0.57,0.71,0.89,1.0,1.14,1.32,1.43,1.57,1.75,1.86,2.0,2.18,2.29,2.43,2.61,2.72},{0.39,0.53,0.71,0.82,0.96,1.14,1.25,1.39,1.57,1.68,1.82,2.0,2.11,2.25,2.43,2.54,2.68,2.86,2.97},{0.69,0.83,1.01,1.12,1.26,1.44,1.55,1.69,1.87,1.98,2.12,2.3,2.41,2.55,2.73,2.84,2.98,3.16,3.27},{0.9,1.04,1.22,1.33,1.47,1.65,1.76,1.9,2.08,2.19,2.33,2.51,2.62,2.76,2.94,3.05,3.19,3.37,3.48},{0.12,0.26,0.44,0.55,0.69,0.87,0.98,1.12,1.3,1.41,1.55,1.73,1.84,1.98,2.16,2.27,2.41,2.59,2.7},{0.92,1.06,1.24,1.35,1.49,1.67,1.78,1.92,2.1,2.21,2.35,2.53,2.64,2.78,2.96,3.07,3.21,3.39,3.5},{0.6,0.74,0.92,1.03,1.17,1.35,1.46,1.6,1.78,1.89,2.03,2.21,2.32,2.46,2.64,2.75,2.89,3.07,3.18},{0.36,0.5,0.68,0.79,0.93,1.11,1.22,1.36,1.54,1.65,1.79,1.97,2.08,2.22,2.4,2.51,2.65,2.83,2.94},{0.48,0.62,0.8,0.91,1.05,1.23,1.34,1.48,1.66,1.77,1.91,2.09,2.2,2.34,2.52,2.63,2.77,2.95,3.06},{0.34,0.48,0.66,0.77,0.91,1.09,1.2,1.34,1.52,1.63,1.77,1.95,2.06,2.2,2.38,2.49,2.63,2.81,2.92},{0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6,2.74,2.92,3.03,3.17,3.35,3.46},{0.09,0.23,0.41,0.52,0.66,0.84,0.95,1.09,1.27,1.38,1.52,1.7,1.81,1.95,2.13,2.24,2.38,2.56,2.67},{0.13,0.27,0.45,0.56,0.7,0.88,0.99,1.13,1.31,1.42,1.56,1.74,1.85,1.99,2.17,2.28,2.42,2.6,2.71},{0.68,0.82,1.0,1.11,1.25,1.43,1.54,1.68,1.86,1.97,2.11,2.29,2.4,2.54,2.72,2.83,2.97,3.15,3.26},{0.41,0.55,0.73,0.84,0.98,1.16,1.27,1.41,1.59,1.7,1.84,2.02,2.13,2.27,2.45,2.56,2.7,2.88,2.99},{0.81,0.95,1.13,1.24,1.38,1.56,1.67,1.81,1.99,2.1,2.24,2.42,2.53,2.67,2.85,2.96,3.1,3.28,3.39},{0.42,0.56,0.74,0.85,0.99,1.17,1.28,1.42,1.6,1.71,1.85,2.03,2.14,2.28,2.46,2.57,2.71,2.89,3.0},{0.84,0.98,1.16,1.27,1.41,1.59,1.7,1.84,2.02,2.13,2.27,2.45,2.56,2.7,2.88,2.99,3.13,3.31,3.42},{0.89,1.03,1.21,1.32,1.46,1.64,1.75,1.89,2.07,2.18,2.32,2.5,2.61,2.75,2.93,3.04,3.18,3.36,3.47},{0.7,0.84,1.02,1.13,1.27,1.45,1.56,1.7,1.88,1.99,2.13,2.31,2.42,2.56,2.74,2.85,2.99,3.17,3.28},{0.28,0.42,0.6,0.71,0.85,1.03,1.14,1.28,1.46,1.57,1.71,1.89,2.0,2.14,2.32,2.43,2.57,2.75,2.86},{0.94,1.08,1.26,1.37,1.51,1.69,1.8,1.94,2.12,2.23,2.37,2.55,2.66,2.8,2.98,3.09,3.23,3.41,3.52},{0.73,0.87,1.05,1.16,1.3,1.48,1.59,1.73,1.91,2.02,2.16,2.34,2.45,2.59,2.77,2.88,3.02,3.2,3.31},{0.92,1.06,1.24,1.35,1.49,1.67,1.78,1.92,2.1,2.21,2.35,2.53,2.64,2.78,2.96,3.07,3.21,3.39,3.5},{0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08,3.22,3.4,3.51},{0.19,0.33,0.51,0.62,0.76,0.94,1.05,1.19,1.37,1.48,1.62,1.8,1.91,2.05,2.23,2.34,2.48,2.66,2.77},{0.14,0.28,0.46,0.57,0.71,0.89,1.0,1.14,1.32,1.43,1.57,1.75,1.86,2.0,2.18,2.29,2.43,2.61,2.72},{0.14,0.28,0.46,0.57,0.71,0.89,1.0,1.14,1.32,1.43,1.57,1.75,1.86,2.0,2.18,2.29,2.43,2.61,2.72},{0.67,0.81,0.99,1.1,1.24,1.42,1.53,1.67,1.85,1.96,2.1,2.28,2.39,2.53,2.71,2.82,2.96,3.14,3.25},{0.2,0.34,0.52,0.63,0.77,0.95,1.06,1.2,1.38,1.49,1.63,1.81,1.92,2.06,2.24,2.35,2.49,2.67,2.78},{0.35,0.49,0.67,0.78,0.92,1.1,1.21,1.35,1.53,1.64,1.78,1.96,2.07,2.21,2.39,2.5,2.64,2.82,2.93},{0.41,0.55,0.73,0.84,0.98,1.16,1.27,1.41,1.59,1.7,1.84,2.02,2.13,2.27,2.45,2.56,2.7,2.88,2.99},{0.12,0.26,0.44,0.55,0.69,0.87,0.98,1.12,1.3,1.41,1.55,1.73,1.84,1.98,2.16,2.27,2.41,2.59,2.7},{0.14,0.28,0.46,0.57,0.71,0.89,1.0,1.14,1.32,1.43,1.57,1.75,1.86,2.0,2.18,2.29,2.43,2.61,2.72},{0.36,0.5,0.68,0.79,0.93,1.11,1.22,1.36,1.54,1.65,1.79,1.97,2.08,2.22,2.4,2.51,2.65,2.83,2.94},{0.53,0.67,0.85,0.96,1.1,1.28,1.39,1.53,1.71,1.82,1.96,2.14,2.25,2.39,2.57,2.68,2.82,3.0,3.11},{0.64,0.78,0.96,1.07,1.21,1.39,1.5,1.64,1.82,1.93,2.07,2.25,2.36,2.5,2.68,2.79,2.93,3.11,3.22},{0.78,0.92,1.1,1.21,1.35,1.53,1.64,1.78,1.96,2.07,2.21,2.39,2.5,2.64,2.82,2.93,3.07,3.25,3.36},{0.6,0.74,0.92,1.03,1.17,1.35,1.46,1.6,1.78,1.89,2.03,2.21,2.32,2.46,2.64,2.75,2.89,3.07,3.18},{0.64,0.78,0.96,1.07,1.21,1.39,1.5,1.64,1.82,1.93,2.07,2.25,2.36,2.5,2.68,2.79,2.93,3.11,3.22},{0.0,0.14,0.32,0.43,0.57,0.75,0.86,1.0,1.18,1.29,1.43,1.61,1.72,1.86,2.04,2.15,2.29,2.47,2.58},{0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08,3.22,3.4,3.51},{0.39,0.53,0.71,0.82,0.96,1.14,1.25,1.39,1.57,1.68,1.82,2.0,2.11,2.25,2.43,2.54,2.68,2.86,2.97},{0.16,0.3,0.48,0.59,0.73,0.91,1.02,1.16,1.34,1.45,1.59,1.77,1.88,2.02,2.2,2.31,2.45,2.63,2.74},{0.79,0.93,1.11,1.22,1.36,1.54,1.65,1.79,1.97,2.08,2.22,2.4,2.51,2.65,2.83,2.94,3.08,3.26,3.37},{0.22,0.36,0.54,0.65,0.79,0.97,1.08,1.22,1.4,1.51,1.65,1.83,1.94,2.08,2.26,2.37,2.51,2.69,2.8},{0.46,0.6,0.78,0.89,1.03,1.21,1.32,1.46,1.64,1.75,1.89,2.07,2.18,2.32,2.5,2.61,2.75,2.93,3.04},{0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08,3.22,3.4,3.51},{0.41,0.55,0.73,0.84,0.98,1.16,1.27,1.41,1.59,1.7,1.84,2.02,2.13,2.27,2.45,2.56,2.7,2.88,2.99},{0.61,0.75,0.93,1.04,1.18,1.36,1.47,1.61,1.79,1.9,2.04,2.22,2.33,2.47,2.65,2.76,2.9,3.08,3.19},{0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08,3.22,3.4,3.51},{0.28,0.42,0.6,0.71,0.85,1.03,1.14,1.28,1.46,1.57,1.71,1.89,2.0,2.14,2.32,2.43,2.57,2.75,2.86},{0.77,0.91,1.09,1.2,1.34,1.52,1.63,1.77,1.95,2.06,2.2,2.38,2.49,2.63,2.81,2.92,3.06,3.24,3.35},{0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08,3.22,3.4,3.51},{0.16,0.3,0.48,0.59,0.73,0.91,1.02,1.16,1.34,1.45,1.59,1.77,1.88,2.02,2.2,2.31,2.45,2.63,2.74},{0.27,0.41,0.59,0.7,0.84,1.02,1.13,1.27,1.45,1.56,1.7,1.88,1.99,2.13,2.31,2.42,2.56,2.74,2.85},{0.0,0.14,0.32,0.43,0.57,0.75,0.86,1.0,1.18,1.29,1.43,1.61,1.72,1.86,2.04,2.15,2.29,2.47,2.58},{0.09,0.23,0.41,0.52,0.66,0.84,0.95,1.09,1.27,1.38,1.52,1.7,1.81,1.95,2.13,2.24,2.38,2.56,2.67},{0.73,0.87,1.05,1.16,1.3,1.48,1.59,1.73,1.91,2.02,2.16,2.34,2.45,2.59,2.77,2.88,3.02,3.2,3.31},{0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6,2.74,2.92,3.03,3.17,3.35,3.46},{0.53,0.67,0.85,0.96,1.1,1.28,1.39,1.53,1.71,1.82,1.96,2.14,2.25,2.39,2.57,2.68,2.82,3.0,3.11},{0.83,0.97,1.15,1.26,1.4,1.58,1.69,1.83,2.01,2.12,2.26,2.44,2.55,2.69,2.87,2.98,3.12,3.3,3.41},{0.05,0.19,0.37,0.48,0.62,0.8,0.91,1.05,1.23,1.34,1.48,1.66,1.77,1.91,2.09,2.2,2.34,2.52,2.63},{0.02,0.16,0.34,0.45,0.59,0.77,0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6},{0.54,0.68,0.86,0.97,1.11,1.29,1.4,1.54,1.72,1.83,1.97,2.15,2.26,2.4,2.58,2.69,2.83,3.01,3.12},{0.02,0.16,0.34,0.45,0.59,0.77,0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6},{0.54,0.68,0.86,0.97,1.11,1.29,1.4,1.54,1.72,1.83,1.97,2.15,2.26,2.4,2.58,2.69,2.83,3.01,3.12},{0.35,0.49,0.67,0.78,0.92,1.1,1.21,1.35,1.53,1.64,1.78,1.96,2.07,2.21,2.39,2.5,2.64,2.82,2.93},{0.63,0.77,0.95,1.06,1.2,1.38,1.49,1.63,1.81,1.92,2.06,2.24,2.35,2.49,2.67,2.78,2.92,3.1,3.21},{0.5,0.64,0.82,0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08},{0.45,0.59,0.77,0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6,2.74,2.92,3.03},{0.4,0.54,0.72,0.83,0.97,1.15,1.26,1.4,1.58,1.69,1.83,2.01,2.12,2.26,2.44,2.55,2.69,2.87,2.98},{0.97,1.11,1.29,1.4,1.54,1.72,1.83,1.97,2.15,2.26,2.4,2.58,2.69,2.83,3.01,3.12,3.26,3.44,3.55},{0.8,0.94,1.12,1.23,1.37,1.55,1.66,1.8,1.98,2.09,2.23,2.41,2.52,2.66,2.84,2.95,3.09,3.27,3.38},{0.58,0.72,0.9,1.01,1.15,1.33,1.44,1.58,1.76,1.87,2.01,2.19,2.3,2.44,2.62,2.73,2.87,3.05,3.16},{0.11,0.25,0.43,0.54,0.68,0.86,0.97,1.11,1.29,1.4,1.54,1.72,1.83,1.97,2.15,2.26,2.4,2.58,2.69},{0.12,0.26,0.44,0.55,0.69,0.87,0.98,1.12,1.3,1.41,1.55,1.73,1.84,1.98,2.16,2.27,2.41,2.59,2.7},{0.23,0.37,0.55,0.66,0.8,0.98,1.09,1.23,1.41,1.52,1.66,1.84,1.95,2.09,2.27,2.38,2.52,2.7,2.81},{0.38,0.52,0.7,0.81,0.95,1.13,1.24,1.38,1.56,1.67,1.81,1.99,2.1,2.24,2.42,2.53,2.67,2.85,2.96},{0.34,0.48,0.66,0.77,0.91,1.09,1.2,1.34,1.52,1.63,1.77,1.95,2.06,2.2,2.38,2.49,2.63,2.81,2.92},{0.51,0.65,0.83,0.94,1.08,1.26,1.37,1.51,1.69,1.8,1.94,2.12,2.23,2.37,2.55,2.66,2.8,2.98,3.09},{0.34,0.48,0.66,0.77,0.91,1.09,1.2,1.34,1.52,1.63,1.77,1.95,2.06,2.2,2.38,2.49,2.63,2.81,2.92},{0.17,0.31,0.49,0.6,0.74,0.92,1.03,1.17,1.35,1.46,1.6,1.78,1.89,2.03,2.21,2.32,2.46,2.64,2.75},{0.91,1.05,1.23,1.34,1.48,1.66,1.77,1.91,2.09,2.2,2.34,2.52,2.63,2.77,2.95,3.06,3.2,3.38,3.49},{0.23,0.37,0.55,0.66,0.8,0.98,1.09,1.23,1.41,1.52,1.66,1.84,1.95,2.09,2.27,2.38,2.52,2.7,2.81},{0.23,0.37,0.55,0.66,0.8,0.98,1.09,1.23,1.41,1.52,1.66,1.84,1.95,2.09,2.27,2.38,2.52,2.7,2.81},{0.12,0.26,0.44,0.55,0.69,0.87,0.98,1.12,1.3,1.41,1.55,1.73,1.84,1.98,2.16,2.27,2.41,2.59,2.7},{0.92,1.06,1.24,1.35,1.49,1.67,1.78,1.92,2.1,2.21,2.35,2.53,2.64,2.78,2.96,3.07,3.21,3.39,3.5},{0.47,0.61,0.79,0.9,1.04,1.22,1.33,1.47,1.65,1.76,1.9,2.08,2.19,2.33,2.51,2.62,2.76,2.94,3.05},{0.09,0.23,0.41,0.52,0.66,0.84,0.95,1.09,1.27,1.38,1.52,1.7,1.81,1.95,2.13,2.24,2.38,2.56,2.67},{0.81,0.95,1.13,1.24,1.38,1.56,1.67,1.81,1.99,2.1,2.24,2.42,2.53,2.67,2.85,2.96,3.1,3.28,3.39},{0.69,0.83,1.01,1.12,1.26,1.44,1.55,1.69,1.87,1.98,2.12,2.3,2.41,2.55,2.73,2.84,2.98,3.16,3.27},{0.99,1.13,1.31,1.42,1.56,1.74,1.85,1.99,2.17,2.28,2.42,2.6,2.71,2.85,3.03,3.14,3.28,3.46,3.57},{0.63,0.77,0.95,1.06,1.2,1.38,1.49,1.63,1.81,1.92,2.06,2.24,2.35,2.49,2.67,2.78,2.92,3.1,3.21},{0.31,0.45,0.63,0.74,0.88,1.06,1.17,1.31,1.49,1.6,1.74,1.92,2.03,2.17,2.35,2.46,2.6,2.78,2.89},{0.04,0.18,0.36,0.47,0.61,0.79,0.9,1.04,1.22,1.33,1.47,1.65,1.76,1.9,2.08,2.19,2.33,2.51,2.62},{0.63,0.77,0.95,1.06,1.2,1.38,1.49,1.63,1.81,1.92,2.06,2.24,2.35,2.49,2.67,2.78,2.92,3.1,3.21},{0.0,0.14,0.32,0.43,0.57,0.75,0.86,1.0,1.18,1.29,1.43,1.61,1.72,1.86,2.04,2.15,2.29,2.47,2.58},{0.61,0.75,0.93,1.04,1.18,1.36,1.47,1.61,1.79,1.9,2.04,2.22,2.33,2.47,2.65,2.76,2.9,3.08,3.19},{0.04,0.18,0.36,0.47,0.61,0.79,0.9,1.04,1.22,1.33,1.47,1.65,1.76,1.9,2.08,2.19,2.33,2.51,2.62},{0.6,0.74,0.92,1.03,1.17,1.35,1.46,1.6,1.78,1.89,2.03,2.21,2.32,2.46,2.64,2.75,2.89,3.07,3.18},{0.84,0.98,1.16,1.27,1.41,1.59,1.7,1.84,2.02,2.13,2.27,2.45,2.56,2.7,2.88,2.99,3.13,3.31,3.42},{0.19,0.33,0.51,0.62,0.76,0.94,1.05,1.19,1.37,1.48,1.62,1.8,1.91,2.05,2.23,2.34,2.48,2.66,2.77},{0.39,0.53,0.71,0.82,0.96,1.14,1.25,1.39,1.57,1.68,1.82,2.0,2.11,2.25,2.43,2.54,2.68,2.86,2.97},{0.58,0.72,0.9,1.01,1.15,1.33,1.44,1.58,1.76,1.87,2.01,2.19,2.3,2.44,2.62,2.73,2.87,3.05,3.16},{0.77,0.91,1.09,1.2,1.34,1.52,1.63,1.77,1.95,2.06,2.2,2.38,2.49,2.63,2.81,2.92,3.06,3.24,3.35},{0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6,2.74,2.92,3.03,3.17,3.35,3.46},{0.22,0.36,0.54,0.65,0.79,0.97,1.08,1.22,1.4,1.51,1.65,1.83,1.94,2.08,2.26,2.37,2.51,2.69,2.8},{0.76,0.9,1.08,1.19,1.33,1.51,1.62,1.76,1.94,2.05,2.19,2.37,2.48,2.62,2.8,2.91,3.05,3.23,3.34},{0.4,0.54,0.72,0.83,0.97,1.15,1.26,1.4,1.58,1.69,1.83,2.01,2.12,2.26,2.44,2.55,2.69,2.87,2.98},{0.71,0.85,1.03,1.14,1.28,1.46,1.57,1.71,1.89,2.0,2.14,2.32,2.43,2.57,2.75,2.86,3.0,3.18,3.29},{0.04,0.18,0.36,0.47,0.61,0.79,0.9,1.04,1.22,1.33,1.47,1.65,1.76,1.9,2.08,2.19,2.33,2.51,2.62},{0.58,0.72,0.9,1.01,1.15,1.33,1.44,1.58,1.76,1.87,2.01,2.19,2.3,2.44,2.62,2.73,2.87,3.05,3.16},{0.54,0.68,0.86,0.97,1.11,1.29,1.4,1.54,1.72,1.83,1.97,2.15,2.26,2.4,2.58,2.69,2.83,3.01,3.12},{0.08,0.22,0.4,0.51,0.65,0.83,0.94,1.08,1.26,1.37,1.51,1.69,1.8,1.94,2.12,2.23,2.37,2.55,2.66},{0.26,0.4,0.58,0.69,0.83,1.01,1.12,1.26,1.44,1.55,1.69,1.87,1.98,2.12,2.3,2.41,2.55,2.73,2.84},{0.2,0.34,0.52,0.63,0.77,0.95,1.06,1.2,1.38,1.49,1.63,1.81,1.92,2.06,2.24,2.35,2.49,2.67,2.78},{0.42,0.56,0.74,0.85,0.99,1.17,1.28,1.42,1.6,1.71,1.85,2.03,2.14,2.28,2.46,2.57,2.71,2.89,3.0},{0.44,0.58,0.76,0.87,1.01,1.19,1.3,1.44,1.62,1.73,1.87,2.05,2.16,2.3,2.48,2.59,2.73,2.91,3.02},{0.9,1.04,1.22,1.33,1.47,1.65,1.76,1.9,2.08,2.19,2.33,2.51,2.62,2.76,2.94,3.05,3.19,3.37,3.48},{0.49,0.63,0.81,0.92,1.06,1.24,1.35,1.49,1.67,1.78,1.92,2.1,2.21,2.35,2.53,2.64,2.78,2.96,3.07},{0.25,0.39,0.57,0.68,0.82,1.0,1.11,1.25,1.43,1.54,1.68,1.86,1.97,2.11,2.29,2.4,2.54,2.72,2.83},{0.5,0.64,0.82,0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08},{0.83,0.97,1.15,1.26,1.4,1.58,1.69,1.83,2.01,2.12,2.26,2.44,2.55,2.69,2.87,2.98,3.12,3.3,3.41},{0.83,0.97,1.15,1.26,1.4,1.58,1.69,1.83,2.01,2.12,2.26,2.44,2.55,2.69,2.87,2.98,3.12,3.3,3.41},{0.77,0.91,1.09,1.2,1.34,1.52,1.63,1.77,1.95,2.06,2.2,2.38,2.49,2.63,2.81,2.92,3.06,3.24,3.35},{0.96,1.1,1.28,1.39,1.53,1.71,1.82,1.96,2.14,2.25,2.39,2.57,2.68,2.82,3.0,3.11,3.25,3.43,3.54},{0.66,0.8,0.98,1.09,1.23,1.41,1.52,1.66,1.84,1.95,2.09,2.27,2.38,2.52,2.7,2.81,2.95,3.13,3.24},{0.89,1.03,1.21,1.32,1.46,1.64,1.75,1.89,2.07,2.18,2.32,2.5,2.61,2.75,2.93,3.04,3.18,3.36,3.47},{0.8,0.94,1.12,1.23,1.37,1.55,1.66,1.8,1.98,2.09,2.23,2.41,2.52,2.66,2.84,2.95,3.09,3.27,3.38},{0.14,0.28,0.46,0.57,0.71,0.89,1.0,1.14,1.32,1.43,1.57,1.75,1.86,2.0,2.18,2.29,2.43,2.61,2.72},{0.31,0.45,0.63,0.74,0.88,1.06,1.17,1.31,1.49,1.6,1.74,1.92,2.03,2.17,2.35,2.46,2.6,2.78,2.89},{0.67,0.81,0.99,1.1,1.24,1.42,1.53,1.67,1.85,1.96,2.1,2.28,2.39,2.53,2.71,2.82,2.96,3.14,3.25},{0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08,3.22,3.4,3.51},{0.01,0.15,0.33,0.44,0.58,0.76,0.87,1.01,1.19,1.3,1.44,1.62,1.73,1.87,2.05,2.16,2.3,2.48,2.59},{0.82,0.96,1.14,1.25,1.39,1.57,1.68,1.82,2.0,2.11,2.25,2.43,2.54,2.68,2.86,2.97,3.11,3.29,3.4},{0.13,0.27,0.45,0.56,0.7,0.88,0.99,1.13,1.31,1.42,1.56,1.74,1.85,1.99,2.17,2.28,2.42,2.6,2.71},{0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08,3.22,3.4,3.51},{0.45,0.59,0.77,0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6,2.74,2.92,3.03},{0.17,0.31,0.49,0.6,0.74,0.92,1.03,1.17,1.35,1.46,1.6,1.78,1.89,2.03,2.21,2.32,2.46,2.64,2.75},{0.09,0.23,0.41,0.52,0.66,0.84,0.95,1.09,1.27,1.38,1.52,1.7,1.81,1.95,2.13,2.24,2.38,2.56,2.67},{0.07,0.21,0.39,0.5,0.64,0.82,0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65},{0.73,0.87,1.05,1.16,1.3,1.48,1.59,1.73,1.91,2.02,2.16,2.34,2.45,2.59,2.77,2.88,3.02,3.2,3.31},{0.14,0.28,0.46,0.57,0.71,0.89,1.0,1.14,1.32,1.43,1.57,1.75,1.86,2.0,2.18,2.29,2.43,2.61,2.72},{0.02,0.16,0.34,0.45,0.59,0.77,0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6},{0.11,0.25,0.43,0.54,0.68,0.86,0.97,1.11,1.29,1.4,1.54,1.72,1.83,1.97,2.15,2.26,2.4,2.58,2.69},{0.2,0.34,0.52,0.63,0.77,0.95,1.06,1.2,1.38,1.49,1.63,1.81,1.92,2.06,2.24,2.35,2.49,2.67,2.78},{0.71,0.85,1.03,1.14,1.28,1.46,1.57,1.71,1.89,2.0,2.14,2.32,2.43,2.57,2.75,2.86,3.0,3.18,3.29},{0.82,0.96,1.14,1.25,1.39,1.57,1.68,1.82,2.0,2.11,2.25,2.43,2.54,2.68,2.86,2.97,3.11,3.29,3.4},{0.61,0.75,0.93,1.04,1.18,1.36,1.47,1.61,1.79,1.9,2.04,2.22,2.33,2.47,2.65,2.76,2.9,3.08,3.19},{0.66,0.8,0.98,1.09,1.23,1.41,1.52,1.66,1.84,1.95,2.09,2.27,2.38,2.52,2.7,2.81,2.95,3.13,3.24},{0.18,0.32,0.5,0.61,0.75,0.93,1.04,1.18,1.36,1.47,1.61,1.79,1.9,2.04,2.22,2.33,2.47,2.65,2.76},{0.93,1.07,1.25,1.36,1.5,1.68,1.79,1.93,2.11,2.22,2.36,2.54,2.65,2.79,2.97,3.08,3.22,3.4,3.51},{0.38,0.52,0.7,0.81,0.95,1.13,1.24,1.38,1.56,1.67,1.81,1.99,2.1,2.24,2.42,2.53,2.67,2.85,2.96},{0.04,0.18,0.36,0.47,0.61,0.79,0.9,1.04,1.22,1.33,1.47,1.65,1.76,1.9,2.08,2.19,2.33,2.51,2.62},{0.24,0.38,0.56,0.67,0.81,0.99,1.1,1.24,1.42,1.53,1.67,1.85,1.96,2.1,2.28,2.39,2.53,2.71,2.82},{0.68,0.82,1.0,1.11,1.25,1.43,1.54,1.68,1.86,1.97,2.11,2.29,2.4,2.54,2.72,2.83,2.97,3.15,3.26},{0.52,0.66,0.84,0.95,1.09,1.27,1.38,1.52,1.7,1.81,1.95,2.13,2.24,2.38,2.56,2.67,2.81,2.99,3.1},{0.01,0.15,0.33,0.44,0.58,0.76,0.87,1.01,1.19,1.3,1.44,1.62,1.73,1.87,2.05,2.16,2.3,2.48,2.59},{0.58,0.72,0.9,1.01,1.15,1.33,1.44,1.58,1.76,1.87,2.01,2.19,2.3,2.44,2.62,2.73,2.87,3.05,3.16},{0.1,0.24,0.42,0.53,0.67,0.85,0.96,1.1,1.28,1.39,1.53,1.71,1.82,1.96,2.14,2.25,2.39,2.57,2.68},{0.58,0.72,0.9,1.01,1.15,1.33,1.44,1.58,1.76,1.87,2.01,2.19,2.3,2.44,2.62,2.73,2.87,3.05,3.16},{0.14,0.28,0.46,0.57,0.71,0.89,1.0,1.14,1.32,1.43,1.57,1.75,1.86,2.0,2.18,2.29,2.43,2.61,2.72},{0.06,0.2,0.38,0.49,0.63,0.81,0.92,1.06,1.24,1.35,1.49,1.67,1.78,1.92,2.1,2.21,2.35,2.53,2.64},{0.89,1.03,1.21,1.32,1.46,1.64,1.75,1.89,2.07,2.18,2.32,2.5,2.61,2.75,2.93,3.04,3.18,3.36,3.47},{0.87,1.01,1.19,1.3,1.44,1.62,1.73,1.87,2.05,2.16,2.3,2.48,2.59,2.73,2.91,3.02,3.16,3.34,3.45},{0.2,0.34,0.52,0.63,0.77,0.95,1.06,1.2,1.38,1.49,1.63,1.81,1.92,2.06,2.24,2.35,2.49,2.67,2.78},{0.32,0.46,0.64,0.75,0.89,1.07,1.18,1.32,1.5,1.61,1.75,1.93,2.04,2.18,2.36,2.47,2.61,2.79,2.9}};
	public double[] target = {3.04,3.21,2.93,3.21,3.32,3.34,3.13,3.23,2.86,3.11,3.41,3.62,2.84,3.64,3.32,3.08,3.2,3.06,3.6,2.81,2.85,3.4,3.13,3.53,3.14,3.56,3.61,3.42,3.0,3.66,3.45,3.64,3.65,2.91,2.86,2.86,3.39,2.92,3.07,3.13,2.84,2.86,3.08,3.25,3.36,3.5,3.32,3.36,2.72,3.65,3.11,2.88,3.51,2.94,3.18,3.65,3.13,3.33,3.65,3.0,3.49,3.65,2.88,2.99,2.72,2.81,3.45,3.6,3.25,3.55,2.77,2.74,3.26,2.74,3.26,3.07,3.35,3.22,3.17,3.12,3.69,3.52,3.3,2.83,2.84,2.95,3.1,3.06,3.23,3.06,2.89,3.63,2.95,2.95,2.84,3.64,3.19,2.81,3.53,3.41,3.71,3.35,3.03,2.76,3.35,2.72,3.33,2.76,3.32,3.56,2.91,3.11,3.3,3.49,3.6,2.94,3.48,3.12,3.43,2.76,3.3,3.26,2.8,2.98,2.92,3.14,3.16,3.62,3.21,2.97,3.22,3.55,3.55,3.49,3.68,3.38,3.61,3.52,2.86,3.03,3.39,3.65,2.73,3.54,2.85,3.65,3.17,2.89,2.81,2.79,3.45,2.86,2.74,2.83,2.92,3.43,3.54,3.33,3.38,2.9,3.65,3.1,2.76,2.96,3.4,3.24,2.73,3.3,2.82,3.3,2.86,2.78,3.61,3.59,2.92,3.04,};
			double[][] inputsV = {{0.92,1.06,1.24,1.35,1.49,1.67,1.78,1.92,2.1,2.21,2.35,2.53,2.64,2.78,2.96,3.07,3.21,3.39,3.5},{0.28,0.42,0.6,0.71,0.85,1.03,1.14,1.28,1.46,1.57,1.71,1.89,2.0,2.14,2.32,2.43,2.57,2.75,2.86},{0.01,0.15,0.33,0.44,0.58,0.76,0.87,1.01,1.19,1.3,1.44,1.62,1.73,1.87,2.05,2.16,2.3,2.48,2.59},{0.76,0.9,1.08,1.19,1.33,1.51,1.62,1.76,1.94,2.05,2.19,2.37,2.48,2.62,2.8,2.91,3.05,3.23,3.34},{0.48,0.62,0.8,0.91,1.05,1.23,1.34,1.48,1.66,1.77,1.91,2.09,2.2,2.34,2.52,2.63,2.77,2.95,3.06},{0.02,0.16,0.34,0.45,0.59,0.77,0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6},{0.78,0.92,1.1,1.21,1.35,1.53,1.64,1.78,1.96,2.07,2.21,2.39,2.5,2.64,2.82,2.93,3.07,3.25,3.36},{0.85,0.99,1.17,1.28,1.42,1.6,1.71,1.85,2.03,2.14,2.28,2.46,2.57,2.71,2.89,3.0,3.14,3.32,3.43},{0.97,1.11,1.29,1.4,1.54,1.72,1.83,1.97,2.15,2.26,2.4,2.58,2.69,2.83,3.01,3.12,3.26,3.44,3.55},{0.75,0.89,1.07,1.18,1.32,1.5,1.61,1.75,1.93,2.04,2.18,2.36,2.47,2.61,2.79,2.9,3.04,3.22,3.33},{0.45,0.59,0.77,0.88,1.02,1.2,1.31,1.45,1.63,1.74,1.88,2.06,2.17,2.31,2.49,2.6,2.74,2.92,3.03},{0.05,0.19,0.37,0.48,0.62,0.8,0.91,1.05,1.23,1.34,1.48,1.66,1.77,1.91,2.09,2.2,2.34,2.52,2.63},{0.29,0.43,0.61,0.72,0.86,1.04,1.15,1.29,1.47,1.58,1.72,1.9,2.01,2.15,2.33,2.44,2.58,2.76,2.87},{0.48,0.62,0.8,0.91,1.05,1.23,1.34,1.48,1.66,1.77,1.91,2.09,2.2,2.34,2.52,2.63,2.77,2.95,3.06},{0.55,0.69,0.87,0.98,1.12,1.3,1.41,1.55,1.73,1.84,1.98,2.16,2.27,2.41,2.59,2.7,2.84,3.02,3.13},{0.2,0.34,0.52,0.63,0.77,0.95,1.06,1.2,1.38,1.49,1.63,1.81,1.92,2.06,2.24,2.35,2.49,2.67,2.78},{0.44,0.58,0.76,0.87,1.01,1.19,1.3,1.44,1.62,1.73,1.87,2.05,2.16,2.3,2.48,2.59,2.73,2.91,3.02},{0.64,0.78,0.96,1.07,1.21,1.39,1.5,1.64,1.82,1.93,2.07,2.25,2.36,2.5,2.68,2.79,2.93,3.11,3.22},{0.87,1.01,1.19,1.3,1.44,1.62,1.73,1.87,2.05,2.16,2.3,2.48,2.59,2.73,2.91,3.02,3.16,3.34,3.45},{0.75,0.89,1.07,1.18,1.32,1.5,1.61,1.75,1.93,2.04,2.18,2.36,2.47,2.61,2.79,2.9,3.04,3.22,3.33},{0.43,0.57,0.75,0.86,1.0,1.18,1.29,1.43,1.61,1.72,1.86,2.04,2.15,2.29,2.47,2.58,2.72,2.9,3.01},{0.6,0.74,0.92,1.03,1.17,1.35,1.46,1.6,1.78,1.89,2.03,2.21,2.32,2.46,2.64,2.75,2.89,3.07,3.18},{0.66,0.8,0.98,1.09,1.23,1.41,1.52,1.66,1.84,1.95,2.09,2.27,2.38,2.52,2.7,2.81,2.95,3.13,3.24}};
	double[] targetV = {3.64,3.0,2.73,3.48,3.2,2.74,3.5,3.57,3.69,3.47,3.17,2.77,3.01,3.2,3.27,2.92,3.16,3.36,3.59,3.47,3.15,3.32,3.38,3.56,};
			double[][] inputss = {{0.84,0.98,1.16,1.27,1.41,1.59,1.7,1.84,2.02,2.13,2.27,2.45,2.56,2.7,2.88,2.99,3.13,3.31,3.42}};
	public double[][] weights = new double[0][0];			//Weight Matrix(2x2)
	public double learnRate = 0.1;							//The Learning rate of the neural network.
	public double momentum = 0.6;
	
	public int inputNodes = 9;								//Number of input nodes for algorithm to allocate
	public int[][] hiddenNodes = {{2,2,2,2,2,2,2},{2,2,2,2,2},{2,2,2},{2,2,2,2,2,2},{2,2,2},{2,2,2,2,2,2,2},{2,2,2,2,2},{2,2,2,2},{2,2,2,2,2},{2,2,2,2},{2,2,2,2,2},{2,2,2,2},{2,2,2,2,2},{2,2,2,2},{2,2,2,2,2},{2,2,2,2},{2,2,2,2,2},{2,2,2,2,2,2},{2,2,2,2}}; 					//hidden layer matrix(2x2) to represent the number of layers: {},{},... and the number of nodes in each individual layer: ex) {2},{3},...  No hiddenNodes = {{1}}
	public int outputNodes = 1;								//Number of output nodes for the algorithm to allocate
	public int phase = 0;									//Current input matrix that algorithm is evaluating
	public int choose = 0;
	public double netError = 0;								//NetError
	public double error = 0;								//Current error of this phase
	
	public int trainPercent = 70;							//Amount of data to be used for training(Percentage wise)
	public int bufferTime = 5;
	
	public ArrayList<Double> recordErrors = new ArrayList<Double>(); //temp data for error values
	public ArrayList<Double> inputLayer = new ArrayList<Double>();	//input matrix of the current layer (Useful for reading big data input information; when input matrix is not in the source code, but in a seperate file)
	public ArrayList<ArrayList<Node>> hiddenLayer = new ArrayList<ArrayList<Node>>(); //hidden layer matrix(2x2) that consists of the number of layers and the number of nodes in the individual layers. 
	public ArrayList<Node> outputLayer = new ArrayList<Node>(); // output layer matrix for storing the number of final nodes for summation and activation.
	double[] historicalgrad = {0};
	
	int i;
	int j;
	int k;
	File file = new File("NetError.txt");
	File file2 = new File("Output.txt");
	File file1 = new File("Pattern.txt");
	FileWriter writer;
	FileWriter writer2;
	
	long tStartOr;
	long tStart;
	long tEnd;
	
	Random rand = new Random();
	
	int constant = 10;
	int count = 0;
	private double[] gradientS = {0.0};
	private double[] ndS = null;
	private boolean flag = false;
	double[] output;										//Final Output
	//Training T;
	TrainSGD T;
	public static void main(String[] args) throws IOException
	{
		int i;
		int j;
		Backpropagation main = new Backpropagation();
		/*BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        System.out.print("Enter # of InputNodes: ");
        String s = br.readLine();
        main.inputNodes = Integer.parseInt(s);
        System.out.print("Enter # of DataLayers: ");
        s = br.readLine();
        int a = Integer.parseInt(s);
        int b = main.inputNodes;
        main.input = new double[a][b];
        main.target = new double[a];
        main.targetV = new double[a];
        main.inputsV = new double[a][b];
        main.inputss = new double[1][b];
        System.out.println("Enter Data(Training)");
        for(i=0;i<a;i++)
        {
        	System.out.println("DataLayer "+i);
        	for(j=0;j<b;j++)
        	{
        		System.out.print("Data: ");
        		s = br.readLine();
        		main.input[i][j] = Integer.parseInt(s);
        	}
        }
        System.out.println("Enter Data(TargetTraining)");
        for(i=0;i<a;i++)
        {
        	System.out.print("Target: ");
        	s = br.readLine();
    		main.target[i] = Integer.parseInt(s);
        }
        System.out.println("Enter Data(Validation)");
        for(i=0;i<a;i++)
        {
        	System.out.println("DataLayer "+i);
        	for(j=0;j<b;j++)
        	{
        		System.out.print("Data: ");
        		s = br.readLine();
        		main.inputsV[i][j] = Integer.parseInt(s);
        	}
        }
        System.out.println("Enter Data(TargetValidation)");
        for(i=0;i<a;i++)
        {
        	System.out.print("Target: ");
        	s = br.readLine();
    		main.targetV[i] = Integer.parseInt(s);
        }
        
        System.out.println("Enter the Problem");
        for(i=0;i<b;i++)
        {
        	System.out.print("Data: ");
        	s = br.readLine();
    		main.inputss[0][i] = Integer.parseInt(s);
        }*/
        main.tStart = System.currentTimeMillis();
        main.tStartOr = System.currentTimeMillis();
        main.Train();
		 //Warning value considered to be null.
	}
	public Backpropagation(){
	      // creates the file
	     /* try {
	    	  file.createNewFile();
				file.createNewFile();
			    writer = new FileWriter(file); 
			    writer2 = new FileWriter(file2);
	      // creates a FileWriter Object

	      // Writes the content to the file
				
		do{
			writer2.write("------------New Iteration----------\n");
		
		  writer.flush();
	      writer.close();
	      writer2.flush();
	      writer2.close();
	     */
		input = new double[176][9];
		inputsV = new double[23][9];
		target = new double[176];
		targetV = new double[24];
		inputss = new double[1][9];
		try(BufferedReader br = new BufferedReader(new FileReader(file1))){
			String line = "";
			boolean vali =  false;
			boolean train = true;
			boolean test = false;
			boolean targetT = false;
			boolean targetV = false;
			
			int countTL = 0;
			int countVL = 0;
			int countT = 0;
			int countV = 0;
			while((line = br.readLine()) != null){
				if(line.endsWith(","))
				{
					if(train)
					{
						countTL++;
					}else if(vali){
						countVL++;
					}
				}
				else if(line.endsWith("i")){
					train = false;
					vali = true;
				}else if(line.endsWith("t"))
				{
					vali = false;
					test = true;
					train = false;
					countT = 0;
				}else if(line.endsWith("T"))
				{
					targetT = true;
					train = false;
					vali = false;
					test = false;
					targetV = false;
					countT = 0;
				}else if(line.endsWith("V"))
				{
					targetT = false;
					train = false;
					vali = false;
					test = false;
					countT = 0;
					targetV = true;
				}
				else
				{
					if(countT == 9 && !targetV && !targetT)
					{
						countT = 0;
					}
					if(countV == 9 && !targetV && !targetT)
					{
						countV = 0;
					}
					if(train)
					{
					input[countTL][countT] = Double.parseDouble(line);
					countT++;
					}else if(vali){
						inputsV[countVL][countV] = Double.parseDouble(line);
						countV++;
					}else if(test){
						
						inputss[0][countT] = Double.parseDouble(line);
						countT++;
					}else if(targetT)
					{
						target[countT] = Double.parseDouble(line);
						countT++;
					}else if(targetV)
					{
						this.targetV[countT] = Double.parseDouble(line);
						countT++;
					}
					
				}
				
			}
		}catch(IOException e)
		{
			
		}
		//Train();
	}
	public void Train()
	{
		do{
		while(phase<input.length)
		{
			setWeights();
			initInputTrain(); //Set input values into the inputLayer ArrayList from the input matrix
			PassNodes();
			TrainNode();
			removeInput();
			phase++;
			//choose = rand.nextInt(input.length) + 0;
			String s = Double.toString(output[0]);
			//writer2.write("Output: "+output[0]+"\n");
			s = Double.toString(error);
			//writer2.write("Error:"+s+"\n");
		}

		CalcNetError();
		count++;
		String s = Double.toString(netError);
		//writer.write(s+"\n"); 
		System.out.println(netError);
		phase = 0;
		
		Mixinput();
		Validate();
	}while(netError>0.00001);

	
	}
	public void Validate()
	{
			while(phase<inputsV.length)
			{
				//setWeights();
				initInputVali(); //Set input values into the inputLayer ArrayList from the input matrix
				PassNodesV();
				//TrainNode();
				removeInput();
				phase++;
				//System.out.println("Output"+output[0]);
			}
			CalcNetError();
			Mixinput();
			phase = 0;
			//System.out.println("NetError"+netError);
		if(netError>0.001)
		{
			//System.out.println("NetError greater than 1. Training again");
			tEnd = System.currentTimeMillis();
			if((tEnd - tStart)/1000 > bufferTime)
			{
				weights = new double[0][0];
				setWeights();
				tStart = System.currentTimeMillis();
			}
			System.out.println("Validation NetError: "+netError);
		}else{
			System.out.println("Satisfactory");
			phase=0;
			while(phase<input.length)
			{
				//setWeights();
				initInputTest(); //Set input values into the inputLayer ArrayList from the input matrix
				PassNodes();
				//TrainNode();
				removeInput();
				phase++;
				System.out.println("Output: "+output[0]);
			}
			phase=0;
			Test();
		}
	}
	public void Test()
	{
		System.out.println("Testing...");
		
		input = inputss;
		/*for(int i=0;i<input.length;i++)
		{
			for(int j=0;j<input[i].length;j++)
			{
				input[i][j] = inputss[i][j];
			}
		}*/
		
		while(phase<input.length)
		{
			//setWeights();
			initInputTest(); //Set input values into the inputLayer ArrayList from the input matrix
			PassNodes();
			//TrainNode();
			removeInput();
			phase++;
			System.out.println("Output"+output[0]);
		}
		tEnd = System.currentTimeMillis();
		CalcNetError();
		System.out.println("NetError"+netError);
		System.out.println("Time Took: "+(tEnd - tStartOr)/1000+" secs.");
	}
	public void Mixinput()
	{
		int high = input.length-1;
		int low = 0;
		for(i=0;i<input.length;i++)
		{
			Random rand = new Random();
			int num = rand.nextInt((high-low)+1)+low;
			double[] imsi = input[num];
			input[num] = input[i];
			input[i] = imsi;
			double imsi1 = target[num];
			target[num] = target[i];
			target[i] = imsi1;
		}
	}
	
	public int ratioPercent(int input, int percent)
	{
		int percent1 = (int)(Math.round(input*input*(percent/100)*100.0)/100.0);
		return percent1;
	}
	public void setWeights()
	{

		if(weights.length <= 0) // Set the weight matrix based on number of inputNodes, outputNodes, and hiddenNodes
		{
			int i;
			int j;
			if(hiddenNodes[0][0]==1) // If there are no hidden nodes...
			{
				weights = new double[(1+0+1)-1][0]; //Set weights[i] as 1 input layer + 1 output layer - 1 layer since we only need 2 matrix layers weights[2] = {0, 1, 2} weights[2-1] = {0, 1}.
				for(i=0;i<weights.length;i++)
				{
					weights[i] = new double[(inputNodes+1)*outputNodes]; //+1 for the bias value
				}
			}else{ // If there are hidden nodes...
				weights = new double[(1+hiddenNodes.length+1)-1][0]; //Set weights[i] as 1 input layer + number of hidden Layers + 1 output layer  - 1 layer since we only need 2+number of hidden layer matrix layers weights[2] = {0, 1, 2} weights[2-1] = {0, 1}.
				for(i=0;i<weights.length;i++)
				{
					if(i==weights.length-1){
						weights[i] = new double[(hiddenNodes[i-1].length+1)*outputNodes];//+1 for the bias value
					}else{
						if(i-1>=0)
						{
							weights[i] = new double[(hiddenNodes[i-1].length+1)*hiddenNodes[i].length]; //+1 for bias value (Need to edit on this part..)
						}else{
							weights[i] = new double[(inputNodes+1)*hiddenNodes[i].length]; //+1 for bias value
						}
					}
				}
			
			}
			
			for(i=0;i<weights.length;i++){
				for(j=0;j<weights[i].length;j++)
				{
					weights[i][j] = 0.0;
				}
			}
	
			Random rand = new Random();
			double high = 1.000;
			double low = -1.000;
			for(i=0;i<weights.length;i++)
			{
				for(j=0;j<weights[i].length;j++)
				{
						weights[i][j] = roundDouble(high+(rand.nextDouble()*(low-high)),10);
						//weights[i][j] = high+(rand.nextDouble()*(low-high));
				}
			}
		}
	}
	public void initInputTrain() // Set the values to the inputLayer ArrayList from the input matrix
	{

		
			for(i=0;i<input[phase].length;i++)
				inputLayer.add(input[phase][i]);
			/*for(i=0;i<inputLayer.size();i++)
			{
				/*String s = Double.toString(inputLayer.get(i));
				try {
					//writer2.write("Contents of input: "+s+"\n");
					s = Double.toString(target[phase]);
					//writer2.write("Target: "+s+"\n");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					System.out.println("Contents of input: " +s+"\n");
				}
				
			}*/
	}
	public void initInputVali()
	{
		for(i=0;i<inputsV[phase].length;i++)
			inputLayer.add(inputsV[phase][i]);
	}
	public void initInputTest()
	{
		for(i=0;i<input[phase].length;i++)
			inputLayer.add(input[phase][i]);
		for(i=0;i<inputLayer.size();i++)
		{
			//String s = Double.toString(inputLayer.get(i));
			/*try {
				//writer2.write("Contents of inputTest: "+s+"\n");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				System.out.println("Contents of inputTest: " +s+"\n");
			}*/
			System.out.println("Contents of input: " +inputLayer.get(i)+"\n");
		}
	}
	public void removeInput() // Remove the input for new input feed
	{
		int i;
		for(i=0;i<input[0].length;i++)
		{
			inputLayer.remove(0);
		}
	}
	public void PassNodes()
	{
		setNodes();
		passOutput();
		error = CalcError();
		//roundFinal();
		//calcNewWeight();

//		flushOutLayer();
	}
	public void PassNodesV()
	{
		setNodes();
		passOutput();
		error = CalcErrorV();
		//roundFinal();
		//calcNewWeight();

//		flushOutLayer();
		recordErrors.add(error);
	}
	public void TrainNode()
	{
		/*if(phase > 1)
		{
			T = new TrainSGD();
			T.Getdval(ndS);
			T.Getgrad(gradientS);
			T.setLayers(inputLayer, hiddenLayer, outputLayer);
			T.setvalues(error, learnRate, momentum);
			T.setWeights(weights);
			ndS = new double[T.num];
			T.Train(1);
			ndS = T.deltaval;
			for(i=0;i<weights.length;i++)
			{
				for(j=0;j<weights[i].length;j++)
				{
					weights[i][j] = T.weights[i][j];
				}
			}
			gradientS = new double[0];
			recordErrors.add(error);

		}else{*/
			T = new TrainSGD();
			T.GethistoryGrad(historicalgrad);
			T.Getgrad(gradientS);
			T.setLayers(inputLayer, hiddenLayer, outputLayer);
			T.setvalues(error, learnRate, momentum);
			T.setWeights(weights);
			if(phase == 0)
			{
				T.Train(0);
				//----
				gradientS = T.gBatch; //Erase a gradient Batch
			}
			if(!flag)
			{
				T.deltaval = new double[T.num];
				ndS = T.deltaval;
				flag = true;
			}
			T.Getdval(ndS);
			ndS = new double[T.num];
			T.Train(1);
			historicalgrad = T.historicalgrad;
			ndS = T.deltaval;
			weights = T.weights;
			gradientS = new double[0];
			recordErrors.add(error);
			
			
		//}
	}
	public void TestNodes()
	{
		setNodes();
		passOutput();
		//flushOutLayer(); //We are not flushing the layer temporarily
	}
	public void setNodes(){
		int i;
		int j;
		if(hiddenNodes[0][0]==1)
		{
			
				flushOutLayer();
				Node node = setNodes1(0, 0); // fix this output error. It does not make sense/
				outputLayer.add(node);
			
		}else{
			
				flushOutLayer(); // Next edit, we put parameters for flushOutLayer(); Make the function have selective removals.
				for(i=0;i<hiddenNodes.length;i++)// suspicion error
				{
					ArrayList<Node> temp = new ArrayList<Node>(); //Marker for line 313
					for(j=0;j<hiddenNodes[i].length;j++)
					{
						/*Node node = what node..;
						temp.add(node);*/
						Node node = setNodes1(i,j);
						temp.add(node);
					}
					hiddenLayer.add(temp);
				}
				for(i=0;i<outputNodes;i++)
				{
					Node node = setNodes1(hiddenNodes.length, i);
					outputLayer.add(node);
				}
			
		}
	}
	public Node setNodes1(int curLayer, int curNode){
		int i;
		double[] tempIN = null;
		double[] tempW = null;
		Node node = null;
		//pre-set weights and output values 
		if(curLayer==0) //If it is the first hiddenLayer
		{
			tempIN = new double[inputLayer.size()+1];
			tempW = new double[inputLayer.size()+1];
			//double[] tempW = new double[weights[0].length];
			for(i=0;i<=inputLayer.size();i++)
			{
				if(i==inputLayer.size())
				{
					tempIN[i] = 1.0;
					tempW[i] = weights[curLayer][weights[curLayer].length-1-curNode];
				}else{
					tempIN[i] = inputLayer.get(i);
					tempW[i] = weights[curLayer][(i*hiddenNodes[curLayer].length)+curNode];
				}
				
			}
			node = new Node(tempIN, tempW);
		}else if(curLayer > 0 && curLayer<hiddenNodes.length-1){
			tempIN = new double[hiddenLayer.get(curLayer-1).size()+1]; // curLayer should start at 1; then -1 to make hiddenLayer[0]
			tempW = new double[hiddenLayer.get(curLayer-1).size()+1]; // So far correct
			for(i=0;i<=hiddenLayer.get(curLayer-1).size();i++) // -1 is because of the input.
			{
				if(i==hiddenLayer.get(curLayer-1).size())
				{
					tempIN[i] = 1.0;
					tempW[i] = weights[curLayer][weights[curLayer].length-1-curNode]; //???
				}else{
					tempIN[i] = (double)hiddenLayer.get(curLayer-1).get(i).eval;
					tempW[i] = weights[curLayer][(i*hiddenNodes[curLayer].length)+curNode]; //There is error whilst changing Hidden Node
				}
				
			}
			node = new Node(tempIN, tempW);
			
		}else{ //I am very very confused about this..
			tempIN = new double[hiddenLayer.get(curLayer-1).size()+1]; // -1 is for the input.
			tempW = new double[weights[curLayer].length];
			for(i=0;i<=hiddenLayer.get(curLayer-1).size();i++) // Stuck on 359 marker. Bias value
			{

				if(i==hiddenLayer.get(curLayer-1).size())
				{
					tempIN[i] = 1.0;
					tempW[i] = weights[curLayer][i];
				}else{
					tempIN[i] = (double)hiddenLayer.get(curLayer-1).get(i).eval;
					tempW[i] = weights[curLayer][i]; // Based on 1 outputvalue.
				}
			}
			node = new Node(tempIN, tempW);
		}
		return node;
	}
	public void passOutput()
	{
		int i;
		output = new double[outputLayer.size()];
		for(i=0;i<output.length;i++)
		{
			output[i] = outputLayer.get(i).eval * constant;
		}
		
	}
	public void flushOutLayer() //Flushing created output and hidden layers for next evaluation
	{
		int i;
		int size = outputLayer.size();
		for(i=0;i<size;i++)
		{
			outputLayer.remove(0);
		}
		size = hiddenLayer.size();
		for(i=0;i<size;i++)
		{
			hiddenLayer.remove(0);
		}
	}
	public double CalcError()
	{
		double error =  output[0] - target[phase];
		//error = (Math.round(error)*100.000)/100.000;
		this.error = error;
		return error;
	}
	public double CalcErrorV()
	{
		double error =  output[0] - targetV[phase];
		//error = (Math.round(error)*100.000)/100.000;
		this.error = error;
		return error;
	}
	public void CalcNetError() //We calculate the Neterror by retreiving all the recorded errors from: ArrayList<Double> recordErrors
	{
		int i;
		double net = 0;
		for(i=0;i<recordErrors.size();i++)
		{
			net += ((recordErrors.get(i)*recordErrors.get(i)));
		}
		netError = (net/2); //Marker 687
		int size = recordErrors.size();
		for(i=0;i<size;i++)
		{
			recordErrors.remove(0);
		}
		
	}
		
	//We need gradient descent algorithm
	// + Basic optimization algorithm
	public int roundInt(double val, double restrict)
	{
		if(val>=restrict)
		{
			return (int)restrict;
		}else{
		val = (Math.round(val)*100.0)/100.0;
		return (int)val;
		}
	}
	public double roundDouble(double val, double restrict)
	{
		if(val>=restrict)
		{
			return (double)restrict;
		}else{
			val = (double)(Math.round(val*1000d))/1000d;
			return val;
		}
	}
	
	
	
}
