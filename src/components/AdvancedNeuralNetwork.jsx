import React, { useState, useEffect } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Zap,
  Plus,
  Trash2,
  Settings,
} from "lucide-react";

const AdvancedNeuralNetwork = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(1);
  const [speed, setSpeed] = useState(1);
  const [learningRate, setLearningRate] = useState(0.5);
  const [showSettings, setShowSettings] = useState(false);

  // Architecture du réseau
  const [hiddenLayers, setHiddenLayers] = useState([2, 2]);
  const [trainingData, setTrainingData] = useState([
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] },
  ]);

  const [editingData, setEditingData] = useState(null);
  const [network, setNetwork] = useState(null);
  const [predictions, setPredictions] = useState([]);

  const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(Math.min(x, 500), -500)));
  const sigmoidDerivative = (x) => x * (1 - x);

  const initializeNetwork = () => {
    const layers = [2, ...hiddenLayers, 1];
    const weights = [];
    const biases = [];

    for (let i = 0; i < layers.length - 1; i++) {
      const w = Array(layers[i + 1])
        .fill(null)
        .map(() =>
          Array(layers[i])
            .fill(null)
            .map(() => Math.random() * 2 - 1),
        );
      const b = Array(layers[i + 1])
        .fill(null)
        .map(() => Math.random() * 2 - 1);
      weights.push(w);
      biases.push(b);
    }

    return { layers, weights, biases };
  };

  useEffect(() => {
    setNetwork(initializeNetwork());
  }, [hiddenLayers]);

  const forward = (input, net) => {
    if (!net) return { activations: [], output: 0 };

    const activations = [input];
    let current = input;

    for (let i = 0; i < net.weights.length; i++) {
      const next = net.weights[i].map((neuronWeights, j) => {
        const sum =
          neuronWeights.reduce((acc, w, k) => acc + w * current[k], 0) +
          net.biases[i][j];
        return sigmoid(sum);
      });
      activations.push(next);
      current = next;
    }

    return { activations, output: current[0] };
  };

  const backward = (input, target, net) => {
    const { activations } = forward(input, net);
    const newWeights = JSON.parse(JSON.stringify(net.weights));
    const newBiases = JSON.parse(JSON.stringify(net.biases));

    let deltas = [];

    // Calcul du delta de sortie
    const outputError = target - activations[activations.length - 1][0];
    deltas[net.weights.length - 1] = [
      outputError * sigmoidDerivative(activations[activations.length - 1][0]),
    ];

    // Rétropropagation des erreurs
    for (let i = net.weights.length - 2; i >= 0; i--) {
      const layerDeltas = [];
      for (let j = 0; j < activations[i + 1].length; j++) {
        let error = 0;
        for (let k = 0; k < deltas[i + 1].length; k++) {
          error += deltas[i + 1][k] * newWeights[i + 1][k][j];
        }
        layerDeltas.push(error * sigmoidDerivative(activations[i + 1][j]));
      }
      deltas[i] = layerDeltas;
    }

    // Mise à jour des poids et biais
    for (let i = 0; i < newWeights.length; i++) {
      for (let j = 0; j < newWeights[i].length; j++) {
        for (let k = 0; k < newWeights[i][j].length; k++) {
          newWeights[i][j][k] +=
            learningRate * deltas[i][j] * activations[i][k];
        }
        newBiases[i][j] += learningRate * deltas[i][j];
      }
    }

    return { weights: newWeights, biases: newBiases };
  };

  const train = () => {
    if (!network) return;

    let newNetwork = JSON.parse(JSON.stringify(network));
    let totalLoss = 0;

    trainingData.forEach(({ input, output }) => {
      const result = backward(input, output[0], newNetwork);
      newNetwork.weights = result.weights;
      newNetwork.biases = result.biases;

      const prediction = forward(input, newNetwork).output;
      totalLoss += Math.pow(output[0] - prediction, 2);
    });

    setNetwork(newNetwork);
    setLoss(totalLoss / trainingData.length);
    setEpoch((prev) => prev + 1);

    const newPredictions = trainingData.map(
      ({ input }) => forward(input, newNetwork).output,
    );
    setPredictions(newPredictions);
  };

  useEffect(() => {
    let interval;
    if (isTraining && network) {
      interval = setInterval(() => {
        train();
      }, 100 / speed);
    }
    return () => clearInterval(interval);
  }, [isTraining, network, speed, learningRate, trainingData]);

  const reset = () => {
    setIsTraining(false);
    setEpoch(0);
    setLoss(1);
    setNetwork(initializeNetwork());
    setPredictions([]);
  };

  const addHiddenLayer = () => {
    setHiddenLayers([...hiddenLayers, 2]);
  };

  const removeHiddenLayer = (index) => {
    if (hiddenLayers.length > 1) {
      const newLayers = hiddenLayers.filter((_, i) => i !== index);
      setHiddenLayers(newLayers);
    }
  };

  const updateLayerSize = (index, size) => {
    const newLayers = [...hiddenLayers];
    newLayers[index] = Math.max(1, Math.min(10, parseInt(size) || 1));
    setHiddenLayers(newLayers);
  };

  const addTrainingExample = () => {
    setTrainingData([...trainingData, { input: [0, 0], output: [0] }]);
  };

  const removeTrainingExample = (index) => {
    if (trainingData.length > 1) {
      setTrainingData(trainingData.filter((_, i) => i !== index));
    }
  };

  const updateTrainingData = (index, field, subIndex, value) => {
    const newData = [...trainingData];
    const numValue = parseFloat(value) || 0;
    newData[index][field][subIndex] = Math.max(0, Math.min(1, numValue));
    setTrainingData(newData);
  };

  const updateWeight = (layerIndex, neuronIndex, weightIndex, value) => {
    if (!network) return;
    const newNetwork = JSON.parse(JSON.stringify(network));
    newNetwork.weights[layerIndex][neuronIndex][weightIndex] =
      parseFloat(value) || 0;
    setNetwork(newNetwork);
  };

  const updateBias = (layerIndex, neuronIndex, value) => {
    if (!network) return;
    const newNetwork = JSON.parse(JSON.stringify(network));
    newNetwork.biases[layerIndex][neuronIndex] = parseFloat(value) || 0;
    setNetwork(newNetwork);
  };

  const renderNetworkVisualization = () => {
    if (!network) return null;

    const layers = network.layers;
    const maxNeurons = Math.max(...layers);
    const layerSpacing = 300 / (layers.length - 1);
    const svgWidth = 400;
    const svgHeight = 300;

    return (
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-80">
        {/* Dessiner les connexions */}
        {layers.map((layerSize, layerIndex) => {
          if (layerIndex === layers.length - 1) return null;

          const nextLayerSize = layers[layerIndex + 1];
          const x1 = 50 + layerIndex * layerSpacing;
          const x2 = 50 + (layerIndex + 1) * layerSpacing;

          return Array(layerSize)
            .fill(null)
            .map((_, i) => {
              const y1 = svgHeight / 2 - (layerSize - 1) * 20 + i * 40;

              return Array(nextLayerSize)
                .fill(null)
                .map((_, j) => {
                  const y2 = svgHeight / 2 - (nextLayerSize - 1) * 20 + j * 40;
                  const weight = network.weights[layerIndex][j][i];
                  const strokeWidth = Math.abs(weight) * 2;
                  const color = weight > 0 ? "#34d399" : "#f87171";

                  return (
                    <line
                      key={`${layerIndex}-${i}-${j}`}
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={color}
                      strokeWidth={strokeWidth}
                      opacity="0.5"
                    />
                  );
                });
            });
        })}

        {/* Dessiner les neurones */}
        {layers.map((layerSize, layerIndex) => {
          const x = 50 + layerIndex * layerSpacing;
          const colors = ["#60a5fa", "#34d399", "#f472b6"];
          const color =
            layerIndex === 0
              ? colors[0]
              : layerIndex === layers.length - 1
                ? colors[2]
                : colors[1];

          return Array(layerSize)
            .fill(null)
            .map((_, i) => {
              const y = svgHeight / 2 - (layerSize - 1) * 20 + i * 40;

              return (
                <g key={`${layerIndex}-${i}`}>
                  <circle
                    cx={x}
                    cy={y}
                    r="15"
                    fill={color}
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <text
                    x={x}
                    y={y + 5}
                    textAnchor="middle"
                    fill="#fff"
                    fontSize="10"
                    fontWeight="bold"
                  >
                    {layerIndex === 0
                      ? `x${i + 1}`
                      : layerIndex === layers.length - 1
                        ? "y"
                        : `h${i + 1}`}
                  </text>
                </g>
              );
            });
        })}

        {/* Labels des couches */}
        {layers.map((_, layerIndex) => {
          const x = 50 + layerIndex * layerSpacing;
          const label =
            layerIndex === 0
              ? "Entrée"
              : layerIndex === layers.length - 1
                ? "Sortie"
                : `Cachée ${layerIndex}`;

          return (
            <text
              key={`label-${layerIndex}`}
              x={x}
              y={20}
              textAnchor="middle"
              fill="#fff"
              fontSize="12"
              fontWeight="bold"
            >
              {label}
            </text>
          );
        })}
      </svg>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-6">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Zap className="text-yellow-400" size={40} />
            Réseau de Neurones Configurable
          </h1>
          <p className="text-purple-200">
            Configuration et entraînement avancés
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mb-4">
          {/* Configuration de l'architecture */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
            <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
              <Settings size={20} />
              Architecture du réseau
            </h2>

            <div className="space-y-3">
              <div className="text-sm text-purple-200">
                Couches cachées: {hiddenLayers.length}
              </div>

              {hiddenLayers.map((size, index) => (
                <div key={index} className="flex items-center gap-2">
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={size}
                    onChange={(e) => updateLayerSize(index, e.target.value)}
                    className="w-16 px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-center"
                  />
                  <span className="text-white text-sm flex-1">
                    Couche {index + 1}: {size} neurones
                  </span>
                  <button
                    onClick={() => removeHiddenLayer(index)}
                    className="p-1 bg-red-500/20 hover:bg-red-500/30 rounded text-red-300"
                    disabled={hiddenLayers.length === 1}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}

              <button
                onClick={addHiddenLayer}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-green-500/20 hover:bg-green-500/30 text-green-300 rounded-lg text-sm"
              >
                <Plus size={16} />
                Ajouter une couche
              </button>
            </div>

            <div className="mt-4 pt-4 border-t border-white/20">
              <div className="space-y-3">
                <div>
                  <label className="text-white text-sm block mb-1">
                    Taux d'apprentissage
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2"
                    value={learningRate}
                    onChange={(e) =>
                      setLearningRate(parseFloat(e.target.value) || 0.5)
                    }
                    className="w-full px-3 py-1 bg-white/10 border border-white/20 rounded text-white"
                  />
                </div>

                <div>
                  <label className="text-white text-sm block mb-1">
                    Vitesse: {speed}x
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="5"
                    step="0.5"
                    value={speed}
                    onChange={(e) => setSpeed(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Visualisation du réseau */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
            <h2 className="text-lg font-semibold text-white mb-3">
              Visualisation
            </h2>
            {renderNetworkVisualization()}
          </div>

          {/* Métriques */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
            <h2 className="text-lg font-semibold text-white mb-3">
              Métriques d'entraînement
            </h2>

            <div className="space-y-4">
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-purple-300 text-sm mb-1">Époque</div>
                <div className="text-white font-bold text-2xl">{epoch}</div>
              </div>

              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-purple-300 text-sm mb-1">Erreur (MSE)</div>
                <div className="text-white font-bold text-2xl">
                  {loss.toFixed(6)}
                </div>
              </div>

              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-purple-300 text-sm mb-1">Précision</div>
                <div className="text-white font-bold text-2xl">
                  {(
                    (predictions.filter(
                      (p, i) => Math.abs(p - trainingData[i].output[0]) < 0.1,
                    ).length /
                      trainingData.length) *
                    100
                  ).toFixed(0)}
                  %
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-2 mt-4">
              <button
                onClick={() => setIsTraining(!isTraining)}
                className={`flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-semibold transition-all ${
                  isTraining
                    ? "bg-red-500 hover:bg-red-600 text-white"
                    : "bg-green-500 hover:bg-green-600 text-white"
                }`}
              >
                {isTraining ? <Pause size={18} /> : <Play size={18} />}
                {isTraining ? "Pause" : "Démarrer"}
              </button>

              <button
                onClick={reset}
                className="flex items-center justify-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg font-semibold transition-all"
              >
                <RotateCcw size={18} />
                Réinitialiser
              </button>
            </div>
          </div>
        </div>

        {/* Données d'entraînement */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 mb-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold text-white">
              Données d'entraînement
            </h2>
            <button
              onClick={addTrainingExample}
              className="flex items-center gap-2 px-3 py-1 bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 rounded-lg text-sm"
            >
              <Plus size={16} />
              Ajouter
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            {trainingData.map((data, i) => (
              <div
                key={i}
                className="bg-white/5 rounded-lg p-3 border border-white/10"
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-white text-sm font-semibold">
                    Exemple {i + 1}
                  </span>
                  <button
                    onClick={() => removeTrainingExample(i)}
                    className="p-1 text-red-400 hover:bg-red-500/20 rounded"
                    disabled={trainingData.length === 1}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>

                <div className="space-y-2">
                  <div>
                    <label className="text-purple-300 text-xs">Entrée 1</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={data.input[0]}
                      onChange={(e) =>
                        updateTrainingData(i, "input", 0, e.target.value)
                      }
                      className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-sm"
                    />
                  </div>

                  <div>
                    <label className="text-purple-300 text-xs">Entrée 2</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={data.input[1]}
                      onChange={(e) =>
                        updateTrainingData(i, "input", 1, e.target.value)
                      }
                      className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-sm"
                    />
                  </div>

                  <div>
                    <label className="text-green-300 text-xs">
                      Sortie attendue
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={data.output[0]}
                      onChange={(e) =>
                        updateTrainingData(i, "output", 0, e.target.value)
                      }
                      className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-sm"
                    />
                  </div>

                  {predictions[i] !== undefined && (
                    <div className="pt-2 border-t border-white/10">
                      <div className="text-pink-300 text-xs">Prédiction</div>
                      <div
                        className={`font-bold text-sm ${
                          Math.abs(predictions[i] - data.output[0]) < 0.1
                            ? "text-green-400"
                            : "text-red-400"
                        }`}
                      >
                        {predictions[i].toFixed(3)}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Poids et biais */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold text-white">Poids et Biais</h2>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="px-3 py-1 bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded-lg text-sm"
            >
              {showSettings ? "Masquer" : "Afficher"}
            </button>
          </div>

          {showSettings && network && (
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {network.weights.map((layerWeights, layerIndex) => (
                <div key={layerIndex} className="bg-white/5 rounded-lg p-3">
                  <h3 className="text-white font-semibold mb-2 text-sm">
                    Couche {layerIndex + 1} → {layerIndex + 2}
                  </h3>

                  {layerWeights.map((neuronWeights, neuronIndex) => (
                    <div key={neuronIndex} className="mb-3">
                      <div className="text-purple-300 text-xs mb-1">
                        Neurone {neuronIndex + 1}
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {neuronWeights.map((weight, weightIndex) => (
                          <div key={weightIndex}>
                            <label className="text-purple-200 text-xs">
                              w{weightIndex + 1}
                            </label>
                            <input
                              type="number"
                              step="0.1"
                              value={weight.toFixed(3)}
                              onChange={(e) =>
                                updateWeight(
                                  layerIndex,
                                  neuronIndex,
                                  weightIndex,
                                  e.target.value,
                                )
                              }
                              className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-xs"
                            />
                          </div>
                        ))}

                        <div>
                          <label className="text-green-200 text-xs">
                            biais
                          </label>
                          <input
                            type="number"
                            step="0.1"
                            value={network.biases[layerIndex][
                              neuronIndex
                            ].toFixed(3)}
                            onChange={(e) =>
                              updateBias(
                                layerIndex,
                                neuronIndex,
                                e.target.value,
                              )
                            }
                            className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-xs"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdvancedNeuralNetwork;
