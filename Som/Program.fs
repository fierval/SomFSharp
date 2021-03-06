﻿open FSom
open System
open System.Collections.Generic
open System.Linq
open Microsoft.FSharp.Linq
open System.IO


let (|Train|_|) (args : Dictionary<string, string>) =
    let dims () = args.["d"].Split(',') |> Array.map(fun s -> Int32.Parse(s.Trim()))
    let keys = args.Keys
    if (keys.Contains "t") then
        let dms = dims()
        let header = if keys.Contains "header" then Int32.Parse(args.["header"]) else 0
        Some(keys.Contains "n", dms.[0], dms.[1], Int32.Parse(args.["e"].Trim()), args.["if"].Trim(), args.["of"].Trim(), header)
    else 
        None

let (|Test|_|) (args : Dictionary<string, string>) =
    let dims () = args.["d"].Split(',') |> Array.map(fun s -> Int32.Parse(s.Trim()))
    let keys = args.Keys
    if (keys.Contains "r") then
        let dms = dims()
        let header = if keys.Contains "header" then Int32.Parse(args.["header"]) else 0
        Some(args.["if"].Trim(), args.["tf"].Trim(), args.["of"].Trim(), header)
    else 
        None

let (|Normalize|_|) (args : Dictionary<string, string>) =
    let dims () = args.["d"].Split(',') |> Array.map(fun s -> Int32.Parse(s.Trim()))
    let keys = args.Keys
    if (keys.Contains "n" && not(keys.Contains "t") && not (keys.Contains "r")) then
        let dms = dims()
        Some(dms.[0], dms.[1], args.["if"].Trim(), args.["of"].Trim())
    else 
        None

let run (args : Dictionary<string, string>) =
    let keys = args.Keys
    if (keys.Contains "t") && (keys.Contains "r") then failwith "both \"t\" and \"r\" are not allowed." 

    match args with
    | Train (normalize, height, width, epochs, fileName, outFile, header) ->
        let som = SomGpu((height, width), fileName, header)  
        //let som = SomGpuTest((height, width), fileName)  
        if normalize then
            som.NormalizeInput(Normalization.Zscore)
            printfn "Normalized som using z-score normalization"
        som.Train epochs |> ignore

        //let som1 = Som((height, width), fileName)
        //som1.somMap <- som.somMap

        //som.SingleDimTrain som.InputNodes.[0]
        printfn "Finished training..."
        if som.ShouldClassify then
            printfn "Classifier will be trained..."
        printfn "Saving to: %s" outFile
        som.Save epochs outFile
    | Normalize (height, width, inFile, outFile) ->
        if not (File.Exists inFile) then failwith "File does not exist: %s" inFile
        if File.Exists outFile then File.Delete outFile
        let som = SomGpu((height, width), inFile)
        som.NormalizeInput(Normalization.Zscore)
        let output = List<string>()
        for node in som.InputNodes do
            let mutable out = node.Name + "\t" + node.Class
            out <- node |> Seq.fold(fun st v -> st + "\t" + v.ToString()) out 
            output.Add(out)
        File.WriteAllLines(outFile, output)
    | Test (trainedFile, testDataFile, outFile, header) ->
        if not (File.Exists trainedFile) || (not (File.Exists testDataFile)) then failwith "input file(s) missing"
        if String.IsNullOrWhiteSpace(outFile) || String.IsNullOrWhiteSpace(testDataFile) || String.IsNullOrWhiteSpace(trainedFile) then failwith "File name cannot be empty"

        let som, nodes = Som.ReadTestSom trainedFile testDataFile header
        let height, width = som |> Array2D.length1, som|> Array2D.length2
        let somGpu = SomGpu((height, width), nodes)
        
        somGpu.somMap <- som

        if File.Exists outFile then File.Delete outFile
        let classes = somGpu.Classify nodes
        somGpu.SaveClassified nodes classes outFile
    |_ -> failwith "need to specify either \"t\" or \"r\" or \"n\" as the action" 

[<EntryPoint>]
let main argv = 
    let expectedArgs : ArgInfo list =
        [
            {Command = "t"; Description = "Train Som"; Required = false};
            {Command = "n"; Description = "Normalize"; Required = false};
            {Command = "d"; Description = "Dimensions, comma-separted"; Required = false};
            {Command = "r"; Description = "Test"; Required = false};
            {Command = "if"; Description="Input File (input nodes or data produced by a training run)"; Required = false};
            {Command = "tf"; Description="Test File (contains data to test)"; Required = false};
            {Command = "of"; Description="Output File"; Required = false};
            {Command = "e"; Description="Epochs"; Required = false};
            {Command = "header"; Description="lines to skip before reading"; Required = false};
        ]

    let args = ParseArgs argv expectedArgs
    try
        run args
        0
    with 
    | e -> Console.WriteLine(e.ToString()); 1
    

     
