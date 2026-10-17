import axios from 'axios';

/**
 * Groq AI Fetch Utility for Molecular Reasoning
 */
export const fetchAIReasoning = async (params: {
    target: string,
    smiles: string,
    qed: number,
    logp: number,
    tpsa: number,
    mw: number,
    affinity: number,
    sa: number
}) => {
    const apiKey = process.env.NEXT_PUBLIC_GROQ_API_KEY;

    console.log("Fetching AI Reasoning for:", params.smiles, "Target:", params.target);

    if (!apiKey) {
        console.error("GROQ API KEY is MISSING. Please update your .env and restart the dev server.");
        return "⚠️ GROQ_API_KEY is not configured. Please update your .env and restart the dev server.";
    }

    try {
        const response = await axios.post(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                model: "llama-3.3-70b-versatile",
                messages: [
                    {
                        role: "system",
                        content: "You are a professional medicinal chemist AI. Your task is to provide concise, scientifically accurate reasoning for drug candidates."
                    },
                    {
                        role: "user",
                        content: `Explain in exactly 3 bullet points why this molecule is a promising drug candidate for ${params.target}. 
                        Data: SMILES=${params.smiles}, QED=${(params.qed ?? 0).toFixed(2)}, LogP=${(params.logp ?? 0).toFixed(2)}, TPSA=${(params.tpsa ?? 0).toFixed(1)}, MW=${(params.mw ?? 0).toFixed(1)}, BindingAffinity=${(params.affinity ?? 0).toFixed(1)}, SAscore=${(params.sa ?? 0).toFixed(1)}.
                        Format each bullet as: [ICON] Bold claim. Supporting chemical reasoning.
                        Icons to use: 🎯 for target binding, 🛡️ for safety/ADMET, ⚗️ for synthesizability.
                        Keep each bullet under 25 words. Be specific, not generic.`
                    }
                ],
                temperature: 0.5,
                max_tokens: 500
            },
            {
                headers: {
                    "Authorization": `Bearer ${apiKey}`,
                    "Content-Type": "application/json"
                }
            }
        );

        return response.data.choices[0].message.content;
    } catch (error) {
        console.error("Groq AI Error:", error);
        throw error;
    }
};

/**
 * Parses user natural language into a structured CURA command
 */
export const parseCopilotIntent = async (message: string, history: any[]) => {
    const apiKey = process.env.NEXT_PUBLIC_GROQ_API_KEY;

    if (!apiKey) {
        return { intent: "explain", raw_intent_summary: "API Key missing. Please update your .env and restart the dev server." };
    }

    const systemPrompt = `
    You are CURA Copilot, an AI drug discovery assistant. Parse the user's natural language request and extract structured parameters.
   
    Always respond in this exact JSON format (no markdown, no backticks):
    {
      "intent": "generate" | "analyze" | "explain" | "compare" | "navigate" | "refine" | "chat",
      "target": "protein name or null",
      "constraints": {
        "min_qed": number | null,
        "max_logp": number | null,
        "oral_bioavailable": boolean | null,
        "excluded_side_effects": string[],
        "water_soluble": boolean | null,
        "min_sa_score_quality": "easy" | "moderate" | null
      },
      "page": "discovery" | "analysis" | "network" | "similar" | "safety" | null,
      "refinement": "description or null",
      "raw_intent_summary": "one sentence describing what the user wants",
      "direct_response": "your conversational response if intent is 'chat' or 'explain'"
    }

    Rules:
    - If the user asks a general question (e.g., "What is aspirin?", "How are you?"), use intent "chat" and put the answer in "direct_response".
    - If the user wants to use a tool (e.g., "Find a drug", "Go to analysis"), use the appropriate tool intent.
    - Keep "direct_response" concise but scientifically accurate.
    - For tool intents, "raw_intent_summary" should be what the AI says before executing.

    Examples:
    "Find a drug for EGFR lung cancer that won't cause bleeding" →
    { "intent": "generate", "target": "EGFR", "constraints": { "excluded_side_effects": ["bleeding", "hemorrhage"], "oral_bioavailable": true }, "page": "discovery", "refinement": null, "raw_intent_summary": "Finding EGFR candidates excluding bleeding risks." }
    `;

    try {
        const response = await axios.post(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                model: "llama-3.3-70b-versatile",
                messages: [
                    { role: "system", content: systemPrompt },
                    ...history.slice(-6).map(h => ({ role: h.role, content: h.content })),
                    { role: "user", content: message }
                ],
                temperature: 0.2,
                max_tokens: 500,
                response_format: { type: "json_object" }
            },
            {
                headers: {
                    "Authorization": `Bearer ${apiKey}`,
                    "Content-Type": "application/json"
                }
            }
        );

        return JSON.parse(response.data.choices[0].message.content);
    } catch (error) {
        console.error("Copilot Parse Error:", error);
        return { intent: "explain", raw_intent_summary: "I'm sorry, I couldn't parse that request. How can I help?" };
    }
};