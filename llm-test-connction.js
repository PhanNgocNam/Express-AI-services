import { ChatOpenAI } from "@langchain/openai";
// import dotenv from "dotenv";
// dotenv.config();

export async function testLLMConnection() {
    try {
        const llm = new ChatOpenAI({
            model: "gpt-3.5-turbo",
            temperature: 0,
            openAIApiKey: process.env.OPENAI_API_KEY,
        });

        const prompt = "Hello, How is going, bro.";
        const response = await llm.invoke(prompt);
        console.log("OpenAI response:", response);
    } catch (error) {
        console.error("Failed to connect to OpenAI via LangChain:", error);
    }
}
