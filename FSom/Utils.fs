namespace FSom

open System
open System.IO
open System.Diagnostics

open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double
open Microsoft.FSharp.Collections
open System.Collections.Generic
open Microsoft.FSharp.Linq

open System.Linq

[<AutoOpen>]
module Utils =
    let (|IsValid|_|) str =
        match String.IsNullOrEmpty(str) with
        | x when x = true -> None
        | x -> Some str

    let trd = function
        | (x, y, z) -> z

    let read fileName =
        match fileName with
        | IsValid name -> ()
        | _ -> failwith "file name is empty"

        try
            let lines = File.ReadAllLines(fileName)
            seq {
                let dims = ref 0
                let name = ref ""
                let classs = ref ""

                for line in lines do
                    let entries = line.Split([|' '; '\t'|])
                    match entries.Length with
                    | l when l = 2 ->
                        name := entries.[0]
                        classs := entries.[1]
                    | l when l = 1 ->
                        name := entries.[0]
                    | l when l = 0 -> ()
                    | l ->
                        if !dims = 0 || !dims = l then 
                            dims := l 
                            yield (!name, !classs, (entries |> Seq.map (fun e -> Double.Parse(e))).ToList() :> IList<float>) 
                        else
                            failwith "wrong dimensions"
            }
        with
            | e -> Trace.TraceError(e.Message); raise e

    let readAndNormalize fileName (norm : Normalization) =
        let entries = read fileName
        let entriesNormalized = (read fileName).Select(fun e -> trd e ).ToList()
        normalize entriesNormalized norm
        |> Seq.map2 (fun (x, y, z) e2 -> (x, y, e2)) entries

