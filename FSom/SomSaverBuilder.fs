namespace FSom

open System
open System.Linq
open System.Collections.Generic
open System.IO

[<AutoOpen>]
module SaverBuilderModule =
    type SaverBuilder () =
        let buffer = new System.Collections.Generic.List<string>()
        let buildStringSeq arr =
            let height = Array2D.length1 arr
            seq {
                for i = 0 to height - 1 do 
                    yield String (arr.[i..i, 0..] |> Array2D.map(fun a -> a.ToString())
                        |> Seq.cast<string> |> Seq.fold (fun st e -> st + "\t" + e) String.Empty |> Seq.skip 1 |> Seq.toArray)
            }

        member this.Delay f :'a [,] = f()
        member this.Run (x: 'a [,]) = x

        member this.Yield x = 
            Array2D.init 1 1 (fun i j -> x)

        member this.For (seqT, f) =
            seq {
                for s in seqT do
                let x = f s
                yield x
            } |> Seq.last

        [<CustomOperation("add", MaintainsVariableSpace=false)>]
        member this.AddLine (s, line) =
            buffer.Add(line.ToString())
            seq{yield ()}

        [<CustomOperation("separate", MaintainsVariableSpace=false)>]
        member this.SeparateLine (s, line) =
            this.AddLine(s, line) |> ignore
            buffer.Add("==================")
            seq{yield ()}
                    
        [<CustomOperation("write", MaintainsVariableSpace=false)>]
        member this.Write (s, x) =
            buffer.AddRange(x |> buildStringSeq)
            x

        [<CustomOperation("commit", MaintainsVariableSpace=false)>]
        member this.Commit (s, file) =
            if String.IsNullOrWhiteSpace file then failwith "empty file name"
            if File.Exists file then File.Delete file
            File.WriteAllLines(file, buffer)
            buffer.Clear()
            s
    let saver = SaverBuilder()    

    let arr1 = Array2D.init 5 7 (fun i j -> i + j)
    let arr2 = Array2D.init 8 9 (fun i j -> i * j)

    let x = saver {
        write arr1
        separate "Separator"
        write arr2
        commit @"c:\temp\t1.txt"
    }
