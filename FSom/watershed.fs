namespace FSom

open System
open System.Linq
open System.Collections.Generic
open Microsoft.FSharp.Linq
open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic

open MathNet.Numerics.LinearAlgebra.Double
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open System.Linq
open Microsoft.FSharp.Collections
open System.IO

[<AutoOpen>]
module watershed =
   
    type Labels =
    | Unlabeled = -1
    | Watershed = 0

    type Watershed (uMatrix: float [,]) =
        let uMatrix = uMatrix
        let width = uMatrix |> Array2D.length2
        let height = uMatrix |> Array2D.length1

        let labels = Array2D.init height width (fun i j -> int Labels.Unlabeled)
        let minimalpoints = Array2D.zeroCreate height width
        let localmins : (int * int) [] = Array.zeroCreate (height * width) 
        let mutable basins = 0

        /// given a u-matrix retrun a map of its heights
        let getOrderedHeights () =
            let heightMap = Dictionary<float, System.Collections.Generic.List<int*int>>()
            uMatrix 
            |> Array2D.iteri
                (fun i j h -> 
                    if not (heightMap.ContainsKey h) then 
                        heightMap.Add(h, List<int*int>())
                    let lst = heightMap.[h]
                    lst.Add(i,j)
                )
            heightMap.OrderBy(fun kvp -> kvp.Key)

        //neighbors of a point on a rectangular grid
        let getNeighbors (point : int*int) =
            let row, col = fst point, snd point
            let neighbors = List<int*int>()
            for i = -1 to 1 do
                for j = -1 to 1 do
                    if i <> 0 && j <> 0 && i + row >= 0 && j + col >= 0 && i + row < height && j + col < width then
                        neighbors.Add(row + i, col + j)
            neighbors

        let processShed point height =
            let neighbors = getNeighbors point
            
            let availableLabels = HashSet(HashIdentity.Structural)

            let mutable minNeighbor = Unchecked.defaultof<int*int>
            let mutable minNeighborHeight = Double.MaxValue
            let mutable nUnlabeled = 0
            let mutable nBorders = 0

            for neighbor in neighbors do
                let row, col = fst neighbor, snd neighbor
                let marker = labels.[row, col]
                let height = uMatrix.[row, col]
                if marker <> int Labels.Unlabeled && height < minNeighborHeight then
                    minNeighbor <- neighbor
                    minNeighborHeight <- height

                match marker with 
                | m when m = int Labels.Unlabeled -> nUnlabeled <- nUnlabeled + 1
                | n when n = int Labels.Watershed -> nBorders <- nBorders + 1
                | _ -> availableLabels.Add(marker)
            
            let row, col = fst point, snd point
            if availableLabels.Count = 0 then
                if nUnlabeled > 0 then
                    // new basin
                    basins <- basins + 1
                    labels.[row, col] <- basins
                    minimalpoints.[row, col] <- point
                    localmins.[basins] <- point
                else
                    // watershed
                    labels.[row, col] <- int Labels.Watershed
                    minimalpoints.[row, col] <- fst minNeighbor, snd minNeighbor
            elif availableLabels.Count = 1 then
                // belong to an old basin
                let marker = availableLabels.First()
                labels.[row, col] <- marker
                minimalpoints.[row, col] <- fst minNeighbor, snd minNeighbor
            else
                // more than one label, hence - watershed!
                labels.[row, col] <- int Labels.Watershed
                
                
        /// Actually create the watershed
        member this.CreateWatersheds () =
            let heightMap = getOrderedHeights()
            heightMap.Select(
                    fun kvp -> 
                        let height = kvp.Key
                        let points = kvp.Value
                        points |> Seq.iter (fun p -> processShed p height)
            )
   
