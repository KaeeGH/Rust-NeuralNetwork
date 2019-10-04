extern crate rand;

use rand::Rng;
use std::io;

fn ConvertStringInput2Num() -> i32{
	println!("Input 'input'");
    let mut input: String = String::new();
	io::stdin().read_line(&mut input);
	let mut input = input.trim_right().to_owned();
	let num: i32 = input.parse::<i32>().unwrap();
	num
	    
}

struct NeuronAndWeight {
	number: i32,
	I2HWeight: [[f64;4];3],
	H2OWeight: [[f64;3];4],
	InputBias: [f64;4],
    HiddenBias: [f64;3],
    OutputBias: [f64;4],
    TeacherSignal: [f64;4]
}
impl NeuronAndWeight {
	fn new(number: i32, I2HWeight: [[f64;4];3], H2OWeight: [[f64;3];4], InputBias: [f64;4], HiddenBias: [f64;3], OutputBias: [f64;4], TeacherSignal: [f64;4]) -> NeuronAndWeight {
        NeuronAndWeight{
        	number: number,
        	I2HWeight: I2HWeight,
        	H2OWeight: H2OWeight,
        	InputBias: InputBias,
            HiddenBias: HiddenBias,
            OutputBias: OutputBias,
            TeacherSignal: TeacherSignal
        }
	}
	fn inputlayer(&mut self) -> [f64;4] {
		let mut InsideOfInput: [f64;4] =  [0.0;4];
        let neipier: f64 =  2.71;
        let mut x = 0;
        let mut Output: [f64;4] = [0.0;4];
        loop{
            if self.number == 1{
                InsideOfInput[x] = 1.0;
                x = 0;
                break
            }else if self.number % 2 == 0{
                InsideOfInput[x] = 0.0;
                self.number /= 2;
                x += 1;
            }else if self.number % 2 == 1{
                InsideOfInput[x] = 1.0;
                self.number /= 2;
                x += 1;
            }
        }
        for i in 0..4{
            InsideOfInput[i] = InsideOfInput[i] + self.InputBias[i];
        }
        for i in 0..4{
            Output[i] = 1.0 / (1.0 + neipier.powf(InsideOfInput[i] * -1.0));
        }
        Output
	}
	fn Hiddenlayer(&mut self, Input: [f64;4]) -> [f64;3] {
        let mut GInsideOfHidden: [[f64;4];3] = [[0.0;4];3];
        let mut InsideOfHidden: [f64;3] = [0.0;3];
        let neipier: f64 = 2.71;
        let mut Output: [f64;3] = [0.0;3];
        for i in 0..3{
            for j in 0..4{
                GInsideOfHidden[i][j] = self.I2HWeight[i][j] * Input[j];
            }
        }
        for i in 0..3{
            for j in 0..4{
                InsideOfHidden[i] =  InsideOfHidden[i] + GInsideOfHidden[i][j];
            }
            InsideOfHidden[i] = InsideOfHidden[i] + self.HiddenBias[i];
        }
        for i in 0..3{
            Output[i] = 1.0 / (1.0 + neipier.powf(InsideOfHidden[i] * -1.0));
        }
        Output
	}
    fn Outputlayer(&mut self, Hidden: [f64;3]) -> [f64;4] {
        let mut GInsideOfOutput: [[f64;3];4] = [[0.0;3];4];
        let mut InsideOfOutput: [f64;4] = [0.0;4];
        let neipier: f64 = 2.71;
        let mut Output: [f64;4] = [0.0;4];
        for i in 0..4{
            for j in 0..3{
                GInsideOfOutput[i][j] = self.H2OWeight[i][j] * Hidden[j];
            }
        }
        for i in 0..4{
            for j in 0..3{
                InsideOfOutput[i] = InsideOfOutput[i] + GInsideOfOutput[i][j];
            }
            InsideOfOutput[i] =InsideOfOutput[i] + self.OutputBias[i];
        }
        for i in 0..4{
            Output[i] = 1.0 / (1.0 + neipier.powf(InsideOfOutput[i] * -1.0));
        }
        Output
    }
}




//中間層と出力層の結合係数の計算
fn CalculateDeltaWeight1(number: i32, I2HWeight: [[f64;4];3], H2OWeight: [[f64;3];4], InputBias: [f64;4], HiddenBias: [f64;3], OutputBias: [f64;4], TeacherSignal: [f64;4]) -> [[f64;4];3] {
    let mut NeuralNetwork = NeuronAndWeight::new(number, I2HWeight, H2OWeight, InputBias, HiddenBias, OutputBias, TeacherSignal);
    let mut inputlayer = NeuralNetwork.inputlayer();
    let mut Hiddenlayer = NeuralNetwork.Hiddenlayer(inputlayer);
    let mut Outputlayer = NeuralNetwork.Outputlayer(Hiddenlayer);
    let mut DeltaWeight: [[f64;4];3] = [[0.0;4];3];
    for i in 0..3{
        for j in 0..4{
            DeltaWeight[i][j] = -0.19 * (Outputlayer[j] - TeacherSignal[j]) * Outputlayer[j] * (1.0 - Outputlayer[j]) * Hiddenlayer[i];
        }
    }
    DeltaWeight
}
    


//入力層と中間層の結合係数の計算
fn CalculateDeltaWeight2(number: i32, I2HWeight: [[f64;4];3], H2OWeight: [[f64;3];4], InputBias: [f64;4], HiddenBias: [f64;3], OutputBias: [f64;4], TeacherSignal: [f64;4]) -> [[f64;3];4] {
   let mut NeuralNetwork = NeuronAndWeight::new(number, I2HWeight, H2OWeight, InputBias, HiddenBias, OutputBias, TeacherSignal);
   let mut inputlayer = NeuralNetwork.inputlayer();
   let mut Hiddenlayer = NeuralNetwork.Hiddenlayer(inputlayer);
   let mut Outputlayer = NeuralNetwork.Outputlayer(Hiddenlayer);
   let mut DeltaWeight: [[f64;3];4] = [[0.0;3];4];
   let mut Sigma: [f64;4] = [0.0;4];
   let mut sum: f64 = 0.0;
    for i in 0..4{
        for j in 0..3{
            for h in 0..4{
                Sigma[h] = (Outputlayer[h] - TeacherSignal[h]) * Outputlayer[h] * (1.0 - Outputlayer[h]) * H2OWeight[h][j];
            }
        }
        for h in 0..4{
            sum = sum + Sigma[h];
        }
        for j in 0..3{
            DeltaWeight[i][j] = -0.23 * sum * Hiddenlayer[j] * (1.0 - Hiddenlayer[j]) * inputlayer[i];
        }
        sum = 0.0;
    }
    DeltaWeight
}




fn main() {
    println!("Hello, world!");
    let input: i32 = rand::thread_rng().gen_range(1, 16);
    println!("{:?}", input);
    let mut I2HWeight: [[f64;4];3] = [[0.0;4];3];
    let mut H2OWeight: [[f64;3];4] = [[0.0;3];4];
    let mut InputBias: [f64;4] = [0.0;4];
    let mut HiddenBias: [f64;3] = [0.0;3];
    let mut OutputBias: [f64;4] = [0.0;4];
    let mut TeacherSign: [f64;4] = [1.0, 0.0, 1.0, 0.0];
    let mut Progress = 0;
    let mut DeltaH2OWeight: [[f64;3];4] = [[0.0;3];4];
    for i in 0..3{
        for j in 0..3{
            I2HWeight[i][j] = rand::thread_rng().gen_range(-10, 10) as f64;
        }
    }
    for i in 0..4{
        for j in 0..3{
            H2OWeight[i][j] = rand::thread_rng().gen_range(-10, 10) as f64;
        }
    }
    for i in 0..4{
        InputBias[i] = rand::thread_rng().gen_range(-10, 10) as f64;
    }
    for i in 0..3{
        HiddenBias[i] = rand::thread_rng().gen_range(-10, 10) as f64;
    }
    for i in 0..4{
        OutputBias[i] = rand::thread_rng().gen_range(-10, 10) as f64;
    }
    let mut DeltaI2HWeight = CalculateDeltaWeight1(input, I2HWeight, H2OWeight, InputBias, HiddenBias, OutputBias, TeacherSign);
    let mut FixedWeightI2H: [[f64;4];3] = [[0.0;4];3];
    for i in 0..3{
        for j in 0..4{
            FixedWeightI2H[i][j] = I2HWeight[i][j] - DeltaI2HWeight[i][j];
        }
    }
    loop {
        DeltaI2HWeight = CalculateDeltaWeight1(input, FixedWeightI2H, H2OWeight, InputBias, HiddenBias, OutputBias, TeacherSign);
        for i in 0..3{
            for j in 0..4{
                FixedWeightI2H[i][j] = FixedWeightI2H[i][j] - DeltaI2HWeight[i][j];
            }
        }
        for i in 0..3{
            for j in  0..4{
                //println!("{}", DeltaI2HWeight[i][j].abs());
                if DeltaI2HWeight[i][j].abs() <= 0.000001 {
                    Progress += 1;
                }else if DeltaI2HWeight[i][j] == 0.0 {
                    Progress += 1;
                }
            }
        }
        //println!("{:?}", DeltaI2HWeight);
        println!("{:?}", Progress);
        if Progress == 10{
            println!("First Stage Completed");
            Progress = 0;
            break
        }
        Progress = 0;
    }
    let mut DeltaH2OWeight = CalculateDeltaWeight2(input, FixedWeightI2H, H2OWeight, InputBias, HiddenBias, OutputBias, TeacherSign);
    let mut FixedWeightH2O: [[f64;3];4] = [[0.0;3];4];
    for i in 0..4{
        for j in 0..3{
            FixedWeightH2O[i][j] = H2OWeight[i][j] - DeltaH2OWeight[i][j];
        }
    }
    loop {
        DeltaH2OWeight = CalculateDeltaWeight2(input, FixedWeightI2H, FixedWeightH2O, InputBias, HiddenBias, OutputBias, TeacherSign);
        for i in 0..4{
            for j in 0..3{
                FixedWeightH2O[i][j] = FixedWeightH2O[i][j] - DeltaH2OWeight[i][j];
            }
        }
        for i in 0..4{
            for j in  0..3{
                //println!("{}", DeltaI2HWeight[i][j].abs());
                if DeltaH2OWeight[i][j].abs() <= 0.000001 {
                    Progress += 1;
                }else if DeltaH2OWeight[i][j] == 0.0 {
                    Progress += 1;
                }
            }
        }
        //println!("{:?}", DeltaH2OWeight);
        println!("{:?}", Progress);
        if Progress == 10{
            println!("All Completed");
            Progress = 0;
            break
        }
        Progress = 0;
    }
    //学習済みデータをテスト
    let mut LearnedNetwork = NeuronAndWeight::new(input, FixedWeightI2H, FixedWeightH2O, InputBias, HiddenBias, OutputBias, TeacherSign);
    let mut ILayer = LearnedNetwork.inputlayer();
    let mut HLayer = LearnedNetwork.Hiddenlayer(ILayer);
    let mut OLayer = LearnedNetwork.Outputlayer(HLayer);
    for i in 0..4{
        if OLayer[i] < 0.5{
            OLayer[i] = 0.0;
        }else if OLayer[i] > 0.5{
            OLayer[i] = 1.0;
        }
    }
    println!("{:?}", OLayer);
}
