/* Copyright 2017 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// tslint:disable-next-line:max-line-length
import { Array1D, CheckpointLoader, ENV, Scalar } from 'deeplearn';
import { Cache } from './ModelCache';
const NUM_LAYERS = 4;
const IMAGE_SIZE = 64; //64x64 image for each letter

//var json = require('../embeddings.json');

export class FontModel {
    constructor() {
        //Can change metaData to show which sample character
        this.metaData = 'a';
        this.dimensions = 40;
        //embedLayer 0, set all the things to 0, that is the average letter.
        this.embedLayer = 0; //higher layers -> pixels? lower layers -> semantics? 0-3 (4 layers)
        this.range = 0.5; //0.5 (0) -> 10 (1) -> ~20 (2) -> ~20 (3) Need enough gray but not too much
        this.inferCache = new Cache(this, this.infer);
        this.numberOfValidChars = 62;
        this.multiplierScalar = Scalar.new(255);
        // Set up character ID mapping.
        this.charIdMap = {};
        for (let i = 65; i < 91; i++) {
            this.charIdMap[String.fromCharCode(i)] = i - 65;
        }
        for (let i = 97; i < 123; i++) {
            this.charIdMap[String.fromCharCode(i)] = i - 97 + 26;
        }
        for (let i = 48; i < 58; i++) {
            this.charIdMap[String.fromCharCode(i)] = i - 48 + 52;
        }
    }
    load(cb) {
        // Load the network weights from the learnjs checkpoint zoo.
        const checkpointLoader = new CheckpointLoader('https://storage.googleapis.com/learnjs-data/checkpoint_zoo/fonts/');
        checkpointLoader.getAllVariables().then(vars => {
            this.variables = vars;
            //const font_weights = this.variables[`Stack/fully_connected_0/weights`];
            //console.log(this.variables);

            cb();
        });
    }
    get(id, args, priority) {
        args.push(this.metaData);
        return new Promise((resolve, reject) => {
            args.push(() => resolve());
            this.inferCache.get(id, args);
        });
    }
    init() {
        this.math = ENV.math;
    }
    fixdim(embedding) {
        if (!embedding) {
            return embedding;
        }
        const inputVector = embedding.as1D();
        if (inputVector.shape[0] < this.dimensions) {
            return this.math.concat1D(inputVector, Array1D.zeros([this.dimensions - inputVector.shape[0]]));
        } else if (inputVector.shape[0] > this.dimensions) {
            return this.math.slice1D(inputVector, 0, this.dimensions);
        }
        return inputVector;
    }
    infer(args) {
        const embedding = args[0];
        //console.log(embedding);
        const ctx = args[1];
        const char = args[2];
        //console.log(char);
        const cb = args[3];
        const charId = this.charIdMap[char.charAt(0)];
        if (charId == null) {
            throw (new Error('Invalid character id'));
        }
        const adjusted = this.math.scope(keep => {
            const idx = Array1D.new([charId]);
            const onehotVector = this.math.oneHot(idx, this.numberOfValidChars).as1D();
            const inputChar = this.math.concat1D(Array1D.zeros([40]), onehotVector);
            let lastOutput = inputChar;
            for (let i = 0; i < NUM_LAYERS; i++) {
                const weights = this.variables[`Stack/fully_connected_${i + 1}/weights`];
                const biases = this.variables[`Stack/fully_connected_${i + 1}/biases`];
                console.log(i+ " "+weights.shape[0]);
                if (i == this.embedLayer) {
                    // Apply the requested embedding at the specified layer.
                    let slicedEmbed = embedding;
                    if (slicedEmbed.shape[0] < weights.shape[0]) {
                        slicedEmbed = this.math.concat1D(embedding, Array1D.zeros([weights.shape[0] - slicedEmbed.shape[0]]));
                    }
                    if (slicedEmbed.shape[0] > weights.shape[0]) {
                        slicedEmbed = this.math.slice1D(slicedEmbed, 0, weights.shape[0]);
                    }
                    lastOutput = this.math.add(lastOutput, slicedEmbed);
                    //This separates each of the 1024 dims and displays
                }
                lastOutput =
                    this.math.relu(this.math.add(this.math.vectorTimesMatrix(lastOutput, weights), biases));

                //TODO: Uncomment code below to look at different layers
                /*if(i == this.embedLayer){
                    lastOutput = this.math.concat1D(embedding, Array1D.zeros([lastOutput.shape[0] - embedding.shape[0]]));
                    console.log(lastOutput.dataSync());
                }*/
            }
            const finalWeights = this.variables['fully_connected/weights'];
            const finalBiases = this.variables['fully_connected/biases'];
            // Faster than the sigmoid used to train: just clip to [-1, 1].
            const finalOutput = this.math.clip(this.math.add(this.math.vectorTimesMatrix(lastOutput, finalWeights), finalBiases), -1, 1);
            // Convert the inferred tensor to the proper scaling then draw it.
            const scaled = this.math.scalarTimesArray(this.multiplierScalar, finalOutput);
            return this.math.scalarMinusArray(this.multiplierScalar, scaled);
        });
        const d = adjusted.as3D(IMAGE_SIZE, IMAGE_SIZE, 1);
        d.data().then(values => {
            const imageData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
            let pixelOffset = 0;
            for (let i = 0; i < values.length; i++) {
                const value = values[i];
                imageData.data[pixelOffset++] = value;
                imageData.data[pixelOffset++] = value;
                imageData.data[pixelOffset++] = value;
                imageData.data[pixelOffset++] = 255;
            }
            ctx.putImageData(imageData, 0, 0);
            d.dispose();
            cb();
        });
    }
}
