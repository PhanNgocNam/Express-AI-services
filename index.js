import dotenv from "dotenv";
dotenv.config();
import express from "express";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
    RunnableSequence,
    RunnableBranch,
    RunnableLambda,
} from "@langchain/core/runnables";

/**
 * EXPRESS API WITH LANGCHAIN CHAINS - HEALTH CHECK SYSTEM
 *
 * This application demonstrates how to build an Express API that uses LangChain
 * chains to process health check requests. The system evaluates various system
 * components and returns appropriate HTTP status codes based on the results.
 *
 * LEARNING OBJECTIVES:
 * 1. Understand how chains work in practical API scenarios
 * 2. Learn error handling with chains
 * 3. See how to structure chain-based workflows
 * 4. Practice conditional logic in chains
 *
 * ARCHITECTURE:
 * - Express server with multiple endpoints
 * - LangChain chains for different types of health checks
 * - Conditional routing based on check results
 * - Proper error handling and logging
 */

const app = express();
app.use(express.json());

// Initialize the language model
const llm = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-3.5-turbo",
    temperature: 0.1, // Low temperature for consistent health check results
});

// Create a string parser for clean output
const parser = new StringOutputParser();

/**
 * CHAIN 1: SYSTEM HEALTH ANALYZER
 *
 * This chain analyzes system metrics and determines if the system is healthy.
 * It's a perfect example of how chains can process complex logic step by step.
 *
 * FLOW:
 * 1. Format the health check prompt with system metrics
 * 2. Send to LLM for analysis
 * 3. Parse the response
 * 4. Validate the result format
 */

// Step 1: Create the health analysis prompt
const healthAnalysisPrompt = PromptTemplate.fromTemplate(`
You are a system health analyzer. Analyze the following system metrics and determine if the system is healthy.

System Metrics:
- CPU Usage: {cpuUsage}%
- Memory Usage: {memoryUsage}%
- Disk Usage: {diskUsage}%
- Response Time: {responseTime}ms
- Error Rate: {errorRate}%

Rules for Health Assessment:
- CPU Usage > 80% = UNHEALTHY
- Memory Usage > 85% = UNHEALTHY  
- Disk Usage > 90% = UNHEALTHY
- Response Time > 2000ms = UNHEALTHY
- Error Rate > 5% = UNHEALTHY
- Otherwise = HEALTHY

Respond with exactly one word: either "HEALTHY" or "UNHEALTHY"
`);

// Step 2: Create a validation function that ensures proper response format
const validateHealthResponse = (response) => {
    const trimmed = response.trim().toUpperCase();
    if (trimmed === "HEALTHY" || trimmed === "UNHEALTHY") {
        return trimmed;
    }
    throw new Error(`Invalid health check response: ${response}`);
};

// Step 3: Build the complete health analysis chain
const healthAnalysisChain = RunnableSequence.from([
    healthAnalysisPrompt, // Format the prompt with system metrics
    llm, // Send to language model for analysis
    parser, // Parse the string response
    validateHealthResponse, // Validate the response format
]);

/**
 * CHAIN 2: SERVICE DEPENDENCY CHECKER
 *
 * This chain checks if external service dependencies are available.
 * It demonstrates how chains can handle multiple inputs and conditional logic.
 *
 * FLOW:
 * 1. Format prompt with service status information
 * 2. Analyze service dependencies
 * 3. Determine overall dependency health
 * 4. Return structured result
 */

const dependencyCheckPrompt = PromptTemplate.fromTemplate(`
You are a service dependency analyzer. Check if the system's dependencies are healthy.

Service Dependencies:
{dependencies}

Rules:
- If ANY critical service is down, respond with "DEPENDENCY_FAILURE"
- If all critical services are up but some non-critical are down, respond with "PARTIAL_DEPENDENCY"
- If all services are up, respond with "DEPENDENCIES_HEALTHY"

Respond with exactly one of these three options.
`);

const validateDependencyResponse = (response) => {
    const trimmed = response.trim().toUpperCase();
    const validResponses = [
        "DEPENDENCY_FAILURE",
        "PARTIAL_DEPENDENCY",
        "DEPENDENCIES_HEALTHY",
    ];

    if (validResponses.includes(trimmed)) {
        return trimmed;
    }
    throw new Error(`Invalid dependency check response: ${response}`);
};

const dependencyCheckChain = RunnableSequence.from([
    dependencyCheckPrompt,
    llm,
    parser,
    validateDependencyResponse,
]);

/**
 * CHAIN 3: COMPREHENSIVE HEALTH CHAIN
 *
 * This is where the magic happens! This chain demonstrates advanced chain
 * composition by combining multiple chains and implementing conditional logic.
 *
 * FLOW:
 * 1. Run system health and dependency checks in parallel
 * 2. Combine results using conditional logic
 * 3. Determine final health status
 * 4. Return structured response with reasoning
 */

const comprehensiveHealthChain = RunnableSequence.from([
    // Step 1: Prepare data for parallel processing
    async (input) => {
        // This step demonstrates how to transform input for multiple chains
        const systemMetrics = {
            cpuUsage: input.cpuUsage,
            memoryUsage: input.memoryUsage,
            diskUsage: input.diskUsage,
            responseTime: input.responseTime,
            errorRate: input.errorRate,
        };

        // Format dependencies for the dependency chain
        const dependencyString = input.dependencies
            .map(
                (dep) =>
                    `- ${dep.name}: ${dep.status} (${
                        dep.critical ? "Critical" : "Non-Critical"
                    })`
            )
            .join("\n");

        return {
            systemMetrics,
            dependencies: dependencyString,
            originalInput: input,
        };
    },

    // Step 2: Run both health checks in parallel
    async (data) => {
        try {
            // Run both chains simultaneously for better performance
            const [systemHealth, dependencyHealth] = await Promise.all([
                healthAnalysisChain.invoke(data.systemMetrics),
                dependencyCheckChain.invoke({
                    dependencies: data.dependencies,
                }),
            ]);

            return {
                systemHealth,
                dependencyHealth,
                originalInput: data.originalInput,
            };
        } catch (error) {
            // This is crucial - we need to handle chain failures gracefully
            throw new Error(`Health check failed: ${error.message}`);
        }
    },

    // Step 3: Apply business logic to determine final status
    async (results) => {
        const { systemHealth, dependencyHealth, originalInput } = results;

        // Business logic: determine overall health based on both checks
        let overallHealth = "HEALTHY";
        let reasoning = [];

        // Check system health
        if (systemHealth === "UNHEALTHY") {
            overallHealth = "UNHEALTHY";
            reasoning.push("System metrics indicate unhealthy state");
        } else {
            reasoning.push("System metrics are within healthy ranges");
        }

        // Check dependency health
        if (dependencyHealth === "DEPENDENCY_FAILURE") {
            overallHealth = "UNHEALTHY";
            reasoning.push("Critical dependencies are failing");
        } else if (dependencyHealth === "PARTIAL_DEPENDENCY") {
            reasoning.push("Some non-critical dependencies are down");
        } else {
            reasoning.push("All dependencies are healthy");
        }

        return {
            status: overallHealth,
            systemHealth,
            dependencyHealth,
            reasoning,
            timestamp: new Date().toISOString(),
            checkedComponents: {
                cpu: originalInput.cpuUsage,
                memory: originalInput.memoryUsage,
                disk: originalInput.diskUsage,
                responseTime: originalInput.responseTime,
                errorRate: originalInput.errorRate,
                dependencies: originalInput.dependencies.length,
            },
        };
    },
]);

/**
 * CHAIN 4: CONDITIONAL RESPONSE CHAIN
 *
 * This chain demonstrates how to use RunnableBranch for conditional logic.
 * Based on the health check result, it routes to different response formatters.
 */

const healthyResponseChain = RunnableLambda.from(async (input) => ({
    httpStatus: 200,
    message: "System is healthy",
    details: input,
    recommendations: [],
}));

const unhealthyResponseChain = RunnableLambda.from(async (input) => {
    // Generate specific recommendations based on what's wrong
    const recommendations = [];

    if (input.systemHealth === "UNHEALTHY") {
        recommendations.push("Check system resources and optimize performance");
    }

    if (input.dependencyHealth === "DEPENDENCY_FAILURE") {
        recommendations.push(
            "Investigate and restore critical service dependencies"
        );
    }

    return {
        httpStatus: 500,
        message: "System is unhealthy",
        details: input,
        recommendations,
    };
});

const conditionalResponseChain = RunnableBranch.from([
    // If status is HEALTHY, use healthy response chain
    [(input) => input.status === "HEALTHY", healthyResponseChain],
    // Otherwise, use unhealthy response chain
    unhealthyResponseChain,
]);

/**
 * MASTER CHAIN: COMPLETE HEALTH CHECK WORKFLOW
 *
 * This is the final chain that orchestrates everything together.
 * It's a perfect example of how complex workflows can be built
 * by combining simpler chains.
 */

const masterHealthCheckChain = RunnableSequence.from([
    comprehensiveHealthChain, // Analyze system and dependencies
    conditionalResponseChain, // Format response based on results
]);

/**
 * API ENDPOINTS
 *
 * Now let's create the actual API endpoints that use our chains.
 * Each endpoint demonstrates different aspects of chain usage.
 */

// ENDPOINT 1: Basic Health Check
// This endpoint demonstrates simple chain usage with error handling
app.get("/api/health/basic", async (req, res) => {
    try {
        // Simulate some basic system metrics
        const systemMetrics = {
            cpuUsage: Math.random() * 100, // Random CPU usage
            memoryUsage: Math.random() * 100, // Random memory usage
            diskUsage: Math.random() * 100, // Random disk usage
            responseTime: Math.random() * 3000, // Random response time
            errorRate: Math.random() * 10, // Random error rate
        };

        console.log(
            "🔍 Running basic health check with metrics:",
            systemMetrics
        );

        // Use the health analysis chain
        const healthStatus = await healthAnalysisChain.invoke(systemMetrics);

        // Return appropriate HTTP status based on health
        if (healthStatus === "HEALTHY") {
            res.status(200).json({
                status: "healthy",
                message: "All systems operational",
                metrics: systemMetrics,
                timestamp: new Date().toISOString(),
            });
        } else {
            res.status(500).json({
                status: "unhealthy",
                message: "System health check failed",
                metrics: systemMetrics,
                timestamp: new Date().toISOString(),
            });
        }
    } catch (error) {
        console.error("❌ Health check failed:", error);
        res.status(500).json({
            status: "error",
            message: "Health check system failure",
            error: error.message,
            timestamp: new Date().toISOString(),
        });
    }
});

// ENDPOINT 2: Comprehensive Health Check
// This endpoint shows how to use complex chains with multiple inputs
app.post("/api/health/comprehensive", async (req, res) => {
    try {
        // Get health check parameters from request body
        const {
            cpuUsage = Math.random() * 100,
            memoryUsage = Math.random() * 100,
            diskUsage = Math.random() * 100,
            responseTime = Math.random() * 3000,
            errorRate = Math.random() * 10,
            dependencies = [
                { name: "Database", status: "up", critical: true },
                { name: "Cache", status: "up", critical: true },
                {
                    name: "External API",
                    status: Math.random() > 0.7 ? "down" : "up",
                    critical: false,
                },
            ],
        } = req.body;

        const healthCheckInput = {
            cpuUsage,
            memoryUsage,
            diskUsage,
            responseTime,
            errorRate,
            dependencies,
        };

        console.log("🔍 Running comprehensive health check:", healthCheckInput);

        // Use the master chain for complete health analysis
        const result = await masterHealthCheckChain.invoke(healthCheckInput);

        console.log("✅ Health check completed:", result);

        // Return the response with appropriate HTTP status
        res.status(result.httpStatus).json(result);
    } catch (error) {
        console.error("❌ Comprehensive health check failed:", error);
        res.status(500).json({
            status: "error",
            message: "Comprehensive health check system failure",
            error: error.message,
            timestamp: new Date().toISOString(),
        });
    }
});

// ENDPOINT 3: Dependency Check Only
// This endpoint demonstrates using individual chains from a larger workflow
app.post("/api/health/dependencies", async (req, res) => {
    try {
        const { dependencies } = req.body;

        if (!dependencies || !Array.isArray(dependencies)) {
            return res.status(400).json({
                status: "error",
                message: "Dependencies array is required in request body",
            });
        }

        // Format dependencies for the chain
        const dependencyString = dependencies
            .map(
                (dep) =>
                    `- ${dep.name}: ${dep.status} (${
                        dep.critical ? "Critical" : "Non-Critical"
                    })`
            )
            .join("\n");

        console.log("🔍 Checking dependencies:", dependencyString);

        // Use only the dependency check chain
        const dependencyHealth = await dependencyCheckChain.invoke({
            dependencies: dependencyString,
        });

        const isHealthy = dependencyHealth === "DEPENDENCIES_HEALTHY";

        res.status(isHealthy ? 200 : 500).json({
            status: isHealthy ? "healthy" : "unhealthy",
            dependencyHealth,
            dependencies,
            message: isHealthy
                ? "All dependencies are healthy"
                : "Some dependencies are failing",
            timestamp: new Date().toISOString(),
        });
    } catch (error) {
        console.error("❌ Dependency check failed:", error);
        res.status(500).json({
            status: "error",
            message: "Dependency check system failure",
            error: error.message,
            timestamp: new Date().toISOString(),
        });
    }
});

// ENDPOINT 4: Chain Status Information
// This endpoint provides information about the available chains
app.get("/api/health/chains", (req, res) => {
    res.json({
        availableChains: {
            healthAnalysis: {
                name: "System Health Analysis Chain",
                description:
                    "Analyzes system metrics to determine overall health",
                inputs: [
                    "cpuUsage",
                    "memoryUsage",
                    "diskUsage",
                    "responseTime",
                    "errorRate",
                ],
                outputs: ["HEALTHY", "UNHEALTHY"],
            },
            dependencyCheck: {
                name: "Dependency Check Chain",
                description:
                    "Checks the health of external service dependencies",
                inputs: ["dependencies"],
                outputs: [
                    "DEPENDENCIES_HEALTHY",
                    "PARTIAL_DEPENDENCY",
                    "DEPENDENCY_FAILURE",
                ],
            },
            comprehensive: {
                name: "Comprehensive Health Chain",
                description: "Combines system and dependency health checks",
                inputs: ["systemMetrics", "dependencies"],
                outputs: ["Complete health report with recommendations"],
            },
        },
        chainArchitecture: {
            flow: "Input → Health Analysis → Dependency Check → Conditional Response → Output",
            errorHandling:
                "Each chain step includes error handling and validation",
            performance:
                "Parallel execution where possible for optimal performance",
        },
    });
});

// Root endpoint with API documentation
app.get("/", (req, res) => {
    res.json({
        message: "LangChain Express Health Check API",
        version: "1.0.0",
        endpoints: {
            "GET /api/health/basic":
                "Basic system health check using simple chain",
            "POST /api/health/comprehensive":
                "Comprehensive health check with custom parameters",
            "POST /api/health/dependencies": "Check only service dependencies",
            "GET /api/health/chains": "Information about available chains",
        },
        documentation: {
            chains: "This API demonstrates various LangChain patterns including sequential chains, parallel execution, conditional logic, and error handling",
            usage: "Each endpoint returns 200 for healthy systems and 500 for unhealthy systems",
            learning:
                "Study the code to understand how chains transform data step by step",
        },
    });
});

// Global error handler
app.use((err, req, res, next) => {
    console.error("🚨 Global error handler:", err);
    res.status(500).json({
        status: "error",
        message: "Internal server error",
        error:
            process.env.NODE_ENV === "development"
                ? err.message
                : "Something went wrong",
    });
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
    console.log(`🚀 LangChain Express API running on port ${PORT}`);
    console.log(`📚 API Documentation: http://localhost:${PORT}`);
    console.log(
        `🔍 Basic Health Check: http://localhost:${PORT}/api/health/basic`
    );
    console.log(
        `📊 Chain Information: http://localhost:${PORT}/api/health/chains`
    );
    console.log(
        "\n🎯 Learning Focus: Study how chains process data step by step!"
    );
    // await testLLMConnection();
});

export default app;
