import React, { useState } from 'react';
import { Music, Download, Loader2, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const emotions = [
  { name: 'Joy', color: 'bg-yellow-500', gradient: 'from-yellow-400 to-orange-500', description: 'Upbeat and happy melodies' },
  { name: 'Tension', color: 'bg-red-500', gradient: 'from-red-500 to-pink-600', description: 'Intense and dramatic compositions' },
  { name: 'Sadness', color: 'bg-blue-500', gradient: 'from-blue-400 to-indigo-600', description: 'Melancholic and slow tunes' },
  { name: 'Calm', color: 'bg-green-500', gradient: 'from-green-400 to-emerald-600', description: 'Relaxing and peaceful ambient' },
];

function App() {
  const [selectedEmotion, setSelectedEmotion] = useState(emotions[0]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioData, setAudioData] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);
    setAudioData(null);

    try {
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ emotion: selectedEmotion.name }),
      });

      if (!response.ok) {
        throw new Error('Generation failed');
      }

      const data = await response.json();
      setAudioData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Background Gradients */}
      <div className={`absolute top-0 left-0 w-full h-full opacity-20 transition-all duration-1000 bg-gradient-to-br ${selectedEmotion.gradient}`} />
      
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="z-10 w-full max-w-4xl"
      >
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
            AI Music Generator
          </h1>
          <p className="text-slate-400 text-lg">Create unique compositions based on emotions</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Controls Section */}
          <div className="bg-slate-800/50 backdrop-blur-xl p-8 rounded-3xl border border-slate-700 shadow-2xl">
            <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
              <Sparkles className="text-yellow-400" /> Select Mood
            </h2>
            
            <div className="grid grid-cols-2 gap-4 mb-8">
              {emotions.map((emotion) => (
                <button
                  key={emotion.name}
                  onClick={() => setSelectedEmotion(emotion)}
                  className={`p-4 rounded-xl border transition-all duration-300 text-left relative overflow-hidden group ${
                    selectedEmotion.name === emotion.name 
                      ? `border-${emotion.color.split('-')[1]}-400 bg-slate-700/50` 
                      : 'border-slate-700 hover:border-slate-500 bg-slate-800/30'
                  }`}
                >
                  <div className={`absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity ${emotion.color}`} />
                  <h3 className="font-bold text-lg mb-1">{emotion.name}</h3>
                  <p className="text-xs text-slate-400">{emotion.description}</p>
                </button>
              ))}
            </div>

            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg transition-all duration-300 flex items-center justify-center gap-3 ${
                isGenerating 
                  ? 'bg-slate-700 cursor-not-allowed' 
                  : `bg-gradient-to-r ${selectedEmotion.gradient} hover:scale-[1.02] hover:shadow-xl`
              }`}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="animate-spin" /> Composing...
                </>
              ) : (
                <>
                  <Music /> Generate Track
                </>
              )}
            </button>
          </div>

          {/* Player Section */}
          <div className="bg-slate-800/50 backdrop-blur-xl p-8 rounded-3xl border border-slate-700 shadow-2xl flex flex-col justify-center items-center relative min-h-[400px]">
            {audioData ? (
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="w-full text-center"
              >
                <div className={`w-32 h-32 mx-auto mb-8 rounded-full bg-gradient-to-br ${selectedEmotion.gradient} flex items-center justify-center shadow-2xl animate-pulse`}>
                  <Music size={48} />
                </div>
                
                <h3 className="text-2xl font-bold mb-2">{selectedEmotion.name} Composition</h3>
                <p className="text-slate-400 mb-8">Generated by AI Model</p>

                <div className="p-4 bg-slate-700/50 border border-slate-600 rounded-xl mb-6 text-slate-300 text-sm">
                  <p>MIDI file generated successfully.</p>
                </div>

                <a 
                  href={audioData.url} 
                  download={`ai-music-${selectedEmotion.name}.mid`}
                  className="inline-flex items-center gap-2 text-slate-300 hover:text-white transition-colors bg-slate-700 hover:bg-slate-600 px-6 py-3 rounded-lg font-semibold"
                >
                  <Download size={20} /> Download MIDI
                </a>
              </motion.div>
            ) : (
              <div className="text-center text-slate-500">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full border-2 border-dashed border-slate-600 flex items-center justify-center">
                  <Music size={32} />
                </div>
                <p>Select a mood and click generate to start</p>
              </div>
            )}
            
            {error && (
              <div className="absolute bottom-4 left-4 right-4 p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-200 text-sm text-center">
                {error}
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default App;
