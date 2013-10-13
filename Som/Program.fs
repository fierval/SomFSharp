open FSom
open System
open System.Collections.Generic
open System.Linq
open Microsoft.FSharp.Linq


let (|Train|_|) (args : Dictionary<string, string>) =
    let dims () = args.["d"].Split(',') |> Array.map(fun s -> Int32.Parse(s.Trim()))
    let keys = args.Keys
    if (keys.Contains "t") then
        let dms = dims()
        Some(keys.Contains "n", dms.[0], dms.[1], Int32.Parse(args.["e"].Trim()), args.["if"].Trim(), args.["of"].Trim())
    else 
        None

let (|Test|_|) (args : Dictionary<string, string>) =
    let dims () = args.["d"].Split(',') |> Array.map(fun s -> Int64.Parse(s.Trim()))
    let keys = args.Keys
    if (keys.Contains "r") then
        let dms = dims()
        Some(args.["if"].Trim(), args.["tf"].Trim(), args.["of"].Trim())
    else 
        None

let run (args : Dictionary<string, string>) =
    let keys = args.Keys
    if not (keys.Contains "t") && not (keys.Contains "r") then failwith "need to specify either \"t\" or \"r\r as an action" 
    if (keys.Contains "t") && (keys.Contains "r") then failwith "need to specify either \"t\" or \"r\r as an action" 

    match args with
    | Train (normalize, height, width, epochs, fileName, outFile) ->
        let som = SomGpu((height, width), fileName)  
        if normalize then
            som.NormalizeInput(Normalization.Zscore)
            printfn "Normalized som using z-score normalization"
            som.Train(epochs) |> ignore
            printfn "Finished training..."
            if som.ShouldClassify then
                printfn "Classifier will be trained..."
            printfn "Saving to: %s" outFile
            som.Save epochs outFile
    |_ -> failwith "need to specify either \"t\" or \"r\r as an action" 
    0

[<EntryPoint>]
let main argv = 
    let expectedArgs : ArgInfo list =
        [
            {Command = "t"; Description = "Train Som"; Required = false};
            {Command = "n"; Description = "Normalize"; Required = false};
            {Command = "d"; Description = "Dimensions, comma-separted"; Required = false};
            {Command = "r"; Description = "Test"; Required = false};
            {Command = "if"; Description="Input File"; Required = false};
            {Command = "tf"; Description="Test File"; Required = false};
            {Command = "of"; Description="Output File"; Required = false};
            {Command = "e"; Description="Epochs"; Required = false};

        ]

    let args = ParseArgs argv expectedArgs
    run args
     
